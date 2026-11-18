# High-Level Designs

Write one file per non-trivial feature **before** implementing it:
`docs/hld/NNN-<feature>.md`. Template:

```markdown
# HLD NNN: <feature>

## Goal            what problem, which guide Part, which metric should move
## Non-goals
## Design          components, interfaces touched, data/artifacts produced, config keys
## Alternatives    and why not
## Evaluation plan golden-set segments affected, expected change, latency/cost impact
## Risks
## Test plan       unit (fakes), integration (slow), hand-computed examples
```
