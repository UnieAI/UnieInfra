# Host prefix retention work in progress

The initial prerequisite preserves logical block position and boundary kind
from a pinned Device cache entry through its StoreTicket to Host publication.
Metadata is captured when the copy is scheduled and published only after the
existing completion ACK. Host access epochs still use the Host publication
clock; acquired status is not copied between tiers. No KV tensors, transfer
destinations, reference ownership or model precision change.

Previously Host publication used default position -1 and boundary kind chunk,
so Host policy could not distinguish shallow from deep cached boundaries.
Preserving metadata alone does not change eviction or improve throughput.
It is groundwork for a separately tested prefix-retention policy, not a
replacement success criterion for the requested AgentX 2x SLO capacity.

An end-to-end scheduler transfer test checks absence before ACK and retained
position/endpoint metadata after ACK. Full CPU validation and the actual
retention policy remain in progress. This branch starts from a55127fc and does
not include the ineffective d6db7844 cross-group retention experiment.

The metadata-only build passed 453/453 CPU tests. The subsequent opt-in
`TOKENSPEED_HOST_PREFIX_RETENTION` candidate orders evictable Host children by
descending logical token depth before the existing recency keys. Physical page
IDs are only tie-breakers. Group-specific block indices are converted to token
depth; unknown metadata falls back to depth zero. Same-group placement and
fully-evictable-parent fallback remain unchanged. Default is off.

Tests contrast enabled/disabled eviction and ensure pinned deep pages survive.
This is a workload-sensitive candidate, not a universally optimal policy:
shallow retention can harm temporal locality or leave state without matching
attention, and no future request is guaranteed a resumable boundary. Full
CPU tests of the policy, fixed churn replay, quality and loaded AgentX curves
remain required. Its switch exists for controlled validation, not as the
requested deliverable.
