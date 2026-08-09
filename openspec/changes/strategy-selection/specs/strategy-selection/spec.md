## Purpose

Defines how multiple interchangeable implementations of the same operation coexist in the
engine, how one of them is chosen for a given machine and workload, and how an operator
observes, overrides, and pins that choice.

## ADDED Requirements

### Requirement: Implementations coexist rather than replace

The engine SHALL allow more than one implementation to be registered for the same
operation. Introducing an implementation MUST NOT remove or disable an already registered
one.

#### Scenario: A ported implementation is added

- **WHEN** an implementation is ported from another engine and registered for an operation
  that already has one
- **THEN** both implementations are registered and individually selectable
- **AND** the previously registered implementation remains available on every platform
  where it was available before

#### Scenario: Adding an implementation does not move the default

- **WHEN** a new implementation is registered and the change does not declare a change of
  default
- **THEN** the implementation selected by default for every operation, platform and
  objective is the same one as before the registration

### Requirement: Hardware eligibility is expressed with both bounds

An implementation's declared hardware eligibility SHALL support both a lower and an upper
bound. An implementation that requires instructions absent from a newer architecture MUST
be expressible as ineligible for that architecture.

#### Scenario: An implementation restricted to one architecture family

- **WHEN** an implementation requires instructions present only on a specific architecture
  family and declares that family's lower and upper bounds
- **AND** the engine runs on a newer architecture outside that range
- **THEN** the implementation is excluded from selection
- **AND** it is not selected even when the operator requests it by name (see the
  unsatisfiable-request requirement)

#### Scenario: An open-ended lower bound

- **WHEN** an implementation works on a given architecture and every later one, and
  declares only a lower bound
- **THEN** it is eligible on all architectures at or above that bound, including
  architectures released after it was written

### Requirement: Selection is deterministic and identical across ranks

For a fixed machine, operation, workload shape, and objective, selection SHALL yield the
same implementation on every invocation and on every parallel rank of the same job.

#### Scenario: Repeated selection

- **WHEN** the same operation is selected twice under identical conditions
- **THEN** the same implementation is chosen both times

#### Scenario: Tensor-parallel job

- **WHEN** an operation is selected independently on each rank of a tensor-parallel job
- **THEN** every rank chooses the same implementation
- **AND** selection does not depend on wall-clock time, rank-local state, or measurement
  performed independently per rank

### Requirement: Selection is observable

An operator SHALL be able to determine, without modifying the engine, which implementation
was selected for an operation and why each other registered implementation was not.

#### Scenario: Inspecting a selection

- **WHEN** an operator asks the engine which implementation serves an operation on this
  machine
- **THEN** the engine reports the selected implementation
- **AND** for each implementation that was excluded, reports whether it was excluded by
  hardware eligibility, by input-shape or trait applicability, or by ranking

#### Scenario: A capability gap is distinguishable from a workload choice

- **WHEN** an implementation is unavailable because the hardware does not satisfy its
  declared eligibility
- **THEN** the reported reason identifies hardware eligibility as the cause
- **AND** the report is distinguishable from an implementation that was eligible but
  ranked lower

### Requirement: Selection is overridable, and an unsatisfiable request fails loudly

An operator SHALL be able to force a specific implementation for an operation. When the
requested implementation is not eligible on the current machine or for the current input,
the engine MUST fail with an error naming the request and the reason, and MUST NOT
silently fall back to another implementation.

#### Scenario: A satisfiable override

- **WHEN** an operator forces an implementation that is eligible
- **THEN** that implementation is used regardless of ranking

#### Scenario: An unsatisfiable override

- **WHEN** an operator forces an implementation that is ineligible on this hardware
- **THEN** the engine reports an error naming the requested implementation and the
  eligibility rule it fails
- **AND** no other implementation is used in its place

### Requirement: Measured selection is persistable and pinnable

When the engine chooses an implementation by measuring candidates, the result SHALL be
persistable and re-loadable, so that a later run reproduces the same choice without
re-measuring.

#### Scenario: Reusing a measured result

- **WHEN** measured selection results have been persisted for a machine and workload shape
- **AND** the engine starts again on the same machine with the same registered
  implementations
- **THEN** the persisted choice is used
- **AND** no measurement is performed at startup

#### Scenario: Persisted results no longer apply

- **WHEN** persisted results were produced with a different set of registered
  implementations or on different hardware
- **THEN** the persisted results are not applied silently
- **AND** the engine reports that they were rejected and why

### Requirement: Scheduler policies are selectable like other implementations

Scheduling decisions that have more than one defensible implementation — at minimum the
choice of which request to preempt, the ordering of prefill and decode work within a step,
and how a long prompt is split across steps — SHALL be selectable by name under the same
contract as other implementations.

#### Scenario: Selecting a preemption policy

- **WHEN** more than one preemption policy is registered and the operator selects one by
  name
- **THEN** the scheduler uses that policy
- **AND** the other registered policies remain available for selection

#### Scenario: Scheduler policy selection is rank-identical

- **WHEN** a scheduler policy is selected on each rank of a tensor-parallel job
- **THEN** every rank selects the same policy
- **AND** the resulting scheduling decisions are identical across ranks for identical
  input

#### Scenario: Default scheduler behaviour is unchanged by the seam

- **WHEN** no scheduler policy is explicitly selected
- **THEN** the scheduler behaves as it did before policies became selectable
