use crate::{
    ExecutionWitness,
    recover_block::{UncompressedPublicKey, recover_block_with_public_keys},
    witness_db::WitnessDatabase,
};
use alloc::{
    collections::BTreeMap,
    fmt::Debug,
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use alloy_consensus::{BlockHeader, Header};
use alloy_eips::eip7928::BlockAccessList;
use alloy_primitives::{B256, keccak256};
use reth_chainspec::{EthChainSpec, EthereumHardforks};
use reth_consensus::ConsensusError;
use reth_consensus::{Consensus, HeaderValidator};
use reth_ethereum_consensus::{EthBeaconConsensus, validate_block_post_execution};
use reth_ethereum_primitives::{Block, EthPrimitives, EthereumReceipt};
use reth_evm::{
    ConfigureEvm,
    execute::{BlockExecutionOutput, Executor},
};
use alloy_evm::{Evm, block::BlockExecutor};
use revm_database::{State, states::bundle_state::BundleRetention};
use reth_primitives_traits::{RecoveredBlock, SealedHeader};
use reth_trie_common::{HashedPostState, KeccakKeyHasher};
use tries::{StatelessTrie, StatelessTrieError, default::StatelessSparseTrie};

/// BLOCKHASH ancestor lookup window limit per EVM (number of most recent blocks accessible).
const BLOCKHASH_ANCESTOR_LIMIT: usize = 256;

/// Errors that can occur during stateless validation.
#[derive(Debug, thiserror::Error)]
pub enum StatelessValidationError {
    /// Error when the number of ancestor headers exceeds the limit.
    #[error("ancestor header count ({count}) exceeds limit ({limit})")]
    AncestorHeaderLimitExceeded {
        /// The number of headers provided.
        count: usize,
        /// The limit.
        limit: usize,
    },

    /// Error when the ancestor headers do not form a contiguous chain.
    #[error("invalid ancestor chain")]
    InvalidAncestorChain,

    /// Error when revealing the witness data failed.
    #[error("failed to reveal witness data for pre-state root {pre_state_root}")]
    WitnessRevealFailed {
        /// The pre-state root used for verification.
        pre_state_root: B256,
    },

    /// Error during stateless block execution.
    #[error("stateless block execution failed: {0}")]
    StatelessExecutionFailed(String),

    /// Error during consensus validation of the block.
    #[error("consensus validation failed: {0}")]
    ConsensusValidationFailed(#[from] ConsensusError),

    /// Error during stateless state root calculation.
    #[error("stateless state root calculation failed")]
    StatelessStateRootCalculationFailed,

    /// Error calculating the pre-state root from the witness data.
    #[error("stateless pre-state root calculation failed")]
    StatelessPreStateRootCalculationFailed,

    /// Error when required ancestor headers are missing (e.g., parent header for pre-state root).
    #[error("missing required ancestor headers")]
    MissingAncestorHeader,

    /// Error when deserializing ancestor headers
    #[error("could not deserialize ancestor headers")]
    HeaderDeserializationFailed,

    /// Error when the computed state root does not match the one in the block header.
    #[error("mismatched post-state root: {got}\n {expected}")]
    PostStateRootMismatch {
        /// The computed post-state root
        got: B256,
        /// The expected post-state root; in the block header
        expected: B256,
    },

    /// Error when the computed pre-state root does not match the expected one.
    #[error("mismatched pre-state root: {got} \n {expected}")]
    PreStateRootMismatch {
        /// The computed pre-state root
        got: B256,
        /// The expected pre-state root from the previous block
        expected: B256,
    },

    /// Error during signer recovery.
    #[error("signer recovery failed")]
    SignerRecovery,

    /// Error when signature has non-normalized s value in homestead block.
    #[error("signature s value not normalized for homestead block")]
    HomesteadSignatureNotNormalized,

    /// Custom error.
    #[error("{0}")]
    Custom(&'static str),
}

impl From<StatelessTrieError> for StatelessValidationError {
    fn from(err: StatelessTrieError) -> Self {
        match err {
            StatelessTrieError::WitnessRevealFailed { pre_state_root } => {
                Self::WitnessRevealFailed { pre_state_root }
            }
            StatelessTrieError::StatelessStateRootCalculationFailed => {
                Self::StatelessStateRootCalculationFailed
            }
            StatelessTrieError::StatelessPreStateRootCalculationFailed => {
                Self::StatelessPreStateRootCalculationFailed
            }
            StatelessTrieError::PreStateRootMismatch { got, expected } => {
                Self::PreStateRootMismatch { got, expected }
            }
        }
    }
}

/// Output of successful stateless block validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatelessValidationOutput {
    /// Hash of the validated block.
    pub block_hash: B256,
    /// Execution output produced while validating the block.
    pub execution_output: BlockExecutionOutput<EthereumReceipt>,
    /// Block access list produced during execution, if available.
    pub block_access_list: Option<BlockAccessList>,
}

/// Performs stateless validation of a block using the provided witness data.
pub fn stateless_validation<ChainSpec, E>(
    current_block: Block,
    public_keys: Vec<UncompressedPublicKey>,
    witness: ExecutionWitness,
    chain_spec: Arc<ChainSpec>,
    evm_config: E,
) -> Result<StatelessValidationOutput, StatelessValidationError>
where
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
    E: ConfigureEvm<Primitives = EthPrimitives> + Clone + 'static,
{
    stateless_validation_with_trie::<StatelessSparseTrie, ChainSpec, E>(
        current_block,
        public_keys,
        witness,
        chain_spec,
        evm_config,
    )
}

/// Performs stateless validation of a block using a custom `StatelessTrie` implementation.
pub fn stateless_validation_with_trie<T, ChainSpec, E>(
    current_block: Block,
    public_keys: Vec<UncompressedPublicKey>,
    witness: ExecutionWitness,
    chain_spec: Arc<ChainSpec>,
    evm_config: E,
) -> Result<StatelessValidationOutput, StatelessValidationError>
where
    T: StatelessTrie,
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
    E: ConfigureEvm<Primitives = EthPrimitives> + Clone + 'static,
{
    let recovered_block = recover_block_with_public_keys(current_block, public_keys, &*chain_spec)?;

    stateless_validation_recovered_with_trie::<T, ChainSpec, E>(
        recovered_block,
        witness,
        chain_spec,
        evm_config,
    )
}

/// Performs stateless validation of an already-recovered block.
pub fn stateless_validation_recovered<ChainSpec, E>(
    recovered_block: RecoveredBlock<Block>,
    witness: ExecutionWitness,
    chain_spec: Arc<ChainSpec>,
    evm_config: E,
) -> Result<StatelessValidationOutput, StatelessValidationError>
where
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
    E: ConfigureEvm<Primitives = EthPrimitives> + Clone + 'static,
{
    stateless_validation_recovered_with_trie::<StatelessSparseTrie, ChainSpec, E>(
        recovered_block,
        witness,
        chain_spec,
        evm_config,
    )
}

/// Performs stateless validation of an already-recovered block using a custom `StatelessTrie` implementation.
pub fn stateless_validation_recovered_with_trie<T, ChainSpec, E>(
    current_block: RecoveredBlock<Block>,
    witness: ExecutionWitness,
    chain_spec: Arc<ChainSpec>,
    evm_config: E,
) -> Result<StatelessValidationOutput, StatelessValidationError>
where
    T: StatelessTrie,
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
    E: ConfigureEvm<Primitives = EthPrimitives> + Clone + 'static,
{
    let mut ancestor_headers: Vec<_> = witness
        .headers
        .iter()
        .map(|bytes| {
            let hash = keccak256(bytes);
            alloy_rlp::decode_exact::<Header>(bytes)
                .map(|h| SealedHeader::new(h, hash))
                .map_err(|_| StatelessValidationError::HeaderDeserializationFailed)
        })
        .collect::<Result<_, _>>()?;
    ancestor_headers.sort_by_key(|header| header.number());

    let count = ancestor_headers.len();
    if count > BLOCKHASH_ANCESTOR_LIMIT {
        return Err(StatelessValidationError::AncestorHeaderLimitExceeded {
            count,
            limit: BLOCKHASH_ANCESTOR_LIMIT,
        });
    }

    let ancestor_hashes = compute_ancestor_hashes(&current_block, &ancestor_headers)?;

    let parent = match ancestor_headers.last() {
        Some(prev_header) => prev_header,
        None => return Err(StatelessValidationError::MissingAncestorHeader),
    };

    validate_block_consensus(chain_spec.clone(), &current_block, parent)?;

    let (mut trie, bytecode) = T::new(&witness, parent.state_root)?;

    let db = WitnessDatabase::new(&trie, bytecode, ancestor_hashes);

    let executor = evm_config.executor(db);
    let output = executor
        .execute(&current_block)
        .map_err(|e| StatelessValidationError::StatelessExecutionFailed(e.to_string()))?;

    validate_block_post_execution(&current_block, &chain_spec, &output.result, None)
        .map_err(StatelessValidationError::ConsensusValidationFailed)?;

    let hashed_state = HashedPostState::from_bundle_state::<KeccakKeyHasher>(&output.state.state);
    let state_root = trie.calculate_state_root(hashed_state)?;
    if state_root != current_block.state_root {
        return Err(StatelessValidationError::PostStateRootMismatch {
            got: state_root,
            expected: current_block.state_root,
        });
    }

    Ok(StatelessValidationOutput {
        block_hash: current_block.hash_slow(),
        execution_output: output,
        // TODO: populate once reth mainline wires revm's BAL builder (track
        // https://github.com/paradigmxyz/reth/pull/22881). Requires
        // State::builder().with_bal_builder() + bump_bal_index() + take_built_alloy_bal().
        block_access_list: None,
    })
}

/// Like [`stateless_validation_recovered_with_trie`], but drives the block
/// executor TRANSACTION-BY-TRANSACTION and emits the cumulative L2 state root
/// AFTER each transaction (`tx_roots[i]` = root after tx `i`).
///
/// Position B's settlement chains a `StateDelta R_{k-1} -> R_k` per entry; the
/// prover (P3) needs PROVEN intermediate roots to gate those deltas instead of
/// trusting the composer. Each cross-chain interaction is its own L2 tx, so the
/// per-tx root after the user tx IS the per-pair candidate root the composer
/// posts. (A single tx making several calls maps several entries onto ONE
/// tx-root; the prover telescopes those intra-tx interiors and gates only the
/// tx-boundary roots.)
///
/// SOUNDNESS GUARD: a mid-execution snapshot omits block-level post-execution
/// changes (withdrawals, EIP-7002/7251 requests). For eez-dev those are a
/// structural no-op (no system predeploys, empty withdrawals — see the parity
/// spike), so `R_k(snapshot) == candidate_k(sealed)`. If the block carries
/// non-empty requests/withdrawals the assumption breaks: we return an EMPTY
/// `tx_roots` (the prover then degrades to endpoint-only gating) rather than
/// emit unsound mid-roots.
pub fn stateless_validation_recovered_with_pair_roots<T, ChainSpec, E>(
    current_block: RecoveredBlock<Block>,
    witness: ExecutionWitness,
    chain_spec: Arc<ChainSpec>,
    evm_config: E,
) -> Result<(B256, Vec<B256>, Vec<bool>), StatelessValidationError>
where
    T: StatelessTrie,
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
    E: ConfigureEvm<Primitives = EthPrimitives> + Clone + 'static,
{
    // ── setup IDENTICAL to stateless_validation_recovered_with_trie ──
    let mut ancestor_headers: Vec<_> = witness
        .headers
        .iter()
        .map(|bytes| {
            let hash = keccak256(bytes);
            alloy_rlp::decode_exact::<Header>(bytes)
                .map(|h| SealedHeader::new(h, hash))
                .map_err(|_| StatelessValidationError::HeaderDeserializationFailed)
        })
        .collect::<Result<_, _>>()?;
    ancestor_headers.sort_by_key(|header| header.number());
    let count = ancestor_headers.len();
    if count > BLOCKHASH_ANCESTOR_LIMIT {
        return Err(StatelessValidationError::AncestorHeaderLimitExceeded {
            count,
            limit: BLOCKHASH_ANCESTOR_LIMIT,
        });
    }
    let ancestor_hashes = compute_ancestor_hashes(&current_block, &ancestor_headers)?;
    let parent = match ancestor_headers.last() {
        Some(prev_header) => prev_header,
        None => return Err(StatelessValidationError::MissingAncestorHeader),
    };
    validate_block_consensus(chain_spec.clone(), &current_block, parent)?;

    let (endpoint_trie, bytecode) = T::new(&witness, parent.state_root)?;
    let db = WitnessDatabase::new(&endpoint_trie, bytecode, ancestor_hashes);
    let mut state = State::builder().with_database(db).with_bundle_update().build();

    // ── drive tx-by-tx, snapshotting the cumulative root after each tx ──
    let sealed = current_block.sealed_block();
    let mut executor = evm_config
        .executor_for_block(&mut state, sealed)
        .map_err(|e| StatelessValidationError::StatelessExecutionFailed(format!("{e:?}")))?;
    executor
        .apply_pre_execution_changes()
        .map_err(|e| StatelessValidationError::StatelessExecutionFailed(e.to_string()))?;

    // EIP-2935 / EIP-4788 land their pre-block system writes in `state` via
    // apply_pre_execution_changes (a CALL to the history / beacon-root
    // contracts). Capture the post-pre-execution root NOW so a block with NO
    // txs still reflects them: the per-tx snapshot loop below never runs for an
    // empty block, so the endpoint would otherwise fall back to
    // parent.state_root and silently drop the pre-block write (correct
    // pre-Prague by coincidence — an empty block's root equalled the parent's —
    // but WRONG once EIP-2935 is active and an empty block's root changes).
    let pre_exec_root = {
        executor.evm_mut().db_mut().merge_transitions(BundleRetention::Reverts);
        let bundle = executor.evm().db().bundle_state.clone();
        let (mut snap_trie, _) = T::new(&witness, parent.state_root)?;
        let hashed = HashedPostState::from_bundle_state::<KeccakKeyHasher>(&bundle.state);
        snap_trie.calculate_state_root(hashed)?
    };

    let mut tx_roots: Vec<B256> = Vec::new();
    for tx in current_block.transactions_recovered() {
        executor
            .execute_transaction(tx)
            .map_err(|e| StatelessValidationError::StatelessExecutionFailed(e.to_string()))?;
        // Fold per-tx transitions into the cumulative bundle (Reverts kept so
        // execution continues), snapshot it WITHOUT draining, and re-root from
        // a FRESH trie (calculate_state_root mutates; the whole-block witness
        // is a superset → has every prefix's nodes).
        executor.evm_mut().db_mut().merge_transitions(BundleRetention::Reverts);
        let bundle = executor.evm().db().bundle_state.clone();
        let (mut snap_trie, _) = T::new(&witness, parent.state_root)?;
        let hashed = HashedPostState::from_bundle_state::<KeccakKeyHasher>(&bundle.state);
        tx_roots.push(snap_trie.calculate_state_root(hashed)?);
    }

    let result = executor
        .finish()
        .map_err(|e| StatelessValidationError::StatelessExecutionFailed(e.to_string()))?
        .1;

    // Post-execution consensus validation — receipts root, logs bloom,
    // cumulative gas used, EIP-7685 requests hash — same as the
    // whole-block path above. The state-root chain alone does not cover
    // these header fields; omitting this would let a header lie about
    // them while still "validating".
    validate_block_post_execution(&current_block, &chain_spec, &result, None)
        .map_err(StatelessValidationError::ConsensusValidationFailed)?;

    // Endpoint: the last tx-root MUST equal the header's post-state root. For
    // an empty block (no txs) fall back to the post-pre-execution root, not the
    // parent root, so the EIP-2935 pre-block write is not dropped.
    let final_root = tx_roots.last().copied().unwrap_or(pre_exec_root);
    if final_root != current_block.state_root {
        return Err(StatelessValidationError::PostStateRootMismatch {
            got: final_root,
            expected: current_block.state_root,
        });
    }

    // Per-tx receipt statuses: the consumer (eez-prover) refuses windows
    // whose SYSTEM tx reverted — a reverted-but-sealed system tx passes
    // every calldata-derived gate vacuously (the builder refuses to seal
    // one, but a malicious builder might not).
    let tx_statuses: Vec<bool> = result.receipts.iter().map(|r| r.success).collect();

    // SOUNDNESS GUARD: trust mid-snapshots only when post-execution is a no-op.
    let withdrawals_empty = sealed
        .body()
        .withdrawals
        .as_ref()
        .map_or(true, |w| w.is_empty());
    if !result.requests.is_empty() || !withdrawals_empty {
        return Ok((current_block.hash_slow(), Vec::new(), tx_statuses));
    }

    Ok((current_block.hash_slow(), tx_roots, tx_statuses))
}

fn validate_block_consensus<ChainSpec>(
    chain_spec: Arc<ChainSpec>,
    block: &RecoveredBlock<Block>,
    parent: &SealedHeader<Header>,
) -> Result<(), StatelessValidationError>
where
    ChainSpec: Send + Sync + EthChainSpec<Header = Header> + EthereumHardforks + Debug,
{
    let consensus = EthBeaconConsensus::new(chain_spec);

    consensus.validate_header(block.sealed_header())?;
    consensus.validate_header_against_parent(block.sealed_header(), parent)?;

    consensus.validate_block_pre_execution(block)?;

    Ok(())
}

fn compute_ancestor_hashes(
    current_block: &RecoveredBlock<Block>,
    ancestor_headers: &[SealedHeader],
) -> Result<BTreeMap<u64, B256>, StatelessValidationError> {
    let mut ancestor_hashes = BTreeMap::new();

    let mut child_header = current_block.sealed_header();

    for parent_header in ancestor_headers.iter().rev() {
        let parent_hash = child_header.parent_hash();
        ancestor_hashes.insert(parent_header.number, parent_hash);

        if parent_hash != parent_header.hash() {
            return Err(StatelessValidationError::InvalidAncestorChain);
        }

        if parent_header.number + 1 != child_header.number {
            return Err(StatelessValidationError::InvalidAncestorChain);
        }

        child_header = parent_header
    }

    Ok(ancestor_hashes)
}
