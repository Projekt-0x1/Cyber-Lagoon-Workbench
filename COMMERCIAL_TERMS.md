0x1 Commercial Terms v1.0

These terms apply to commercial use under the `LICENSE` file.

## 1) Definitions

- Licensor: Stefan Buesch / Projekt 0x1.
- Software: this repository and derivatives.
- Attributable Gross Revenue: gross revenue from products/services that
  incorporate, depend on, or are materially enabled by the Software.
- Substrate-Embedded Commercial Use: use of the Software (with or without
  modification) as a component, runtime, or technology layer inside a
  commercial product, service, or internal system whose primary purpose is
  something other than offering substrate functionality to third parties.
  Modifications for fit, configuration, deployment customization, and
  integration do not change this classification.
- Substrate Republication: distribution, hosting, or commercialization of the
  Software (or any rename, fork, rebrand, or repackaging thereof) as a
  substrate, AI runtime, cognitive infrastructure, or substantively similar
  standalone offering, where the Software itself (or a derived version) is the
  commercially provided offering to third parties.

## 2) Commercial Royalty Tracks

### Track A: Substrate-Embedded Commercial Use
- Royalty: 7%
- Threshold: first USD 100,000 Attributable Gross Revenue per calendar year is
  royalty-free.
- Royalty base: Attributable Gross Revenue above USD 100,000 per year.

Track A applies regardless of whether you modified the Software, forked it, or
maintain a private copy. The trigger is how you commercialize, not how you
technically structured your code.

Examples (Track A):
- a dog-training platform using 0x1 as its learning engine
- a code analysis tool using 0x1 as its inference layer
- an internal company tool using 0x1 to power features
- a fork of 0x1 modified for your domain, used inside your own product

Example calculation:
- Revenue = USD 250,000
- Royalty = (250,000 - 100,000) * 7% = USD 10,500

### Track B: Substrate Republication
- Royalty: 10%
- Threshold: none
- Royalty base: all Attributable Gross Revenue from first commercial revenue.

Track B applies when the substrate itself (or a renamed/repackaged version of
it) is what you are commercially offering to others, regardless of whether you
modified the code.

Examples (Track B):
- forking 0x1, renaming it "QuantumCore", and selling QuantumCore licenses
- offering substrate-as-a-service based on a 0x1 fork
- redistributing 0x1 under a new brand for commercial fees
- a hosted "AI runtime" service whose runtime is a renamed 0x1 fork

Example calculation:
- Revenue from your renamed substrate offering = USD 50,000
- Royalty = 50,000 * 10% = USD 5,000

### Boundary clarification

If your product's primary purpose is something other than substrate provision
(Track A), modifications and forks for fit do not push you into Track B.
If your product is substrate provision (Track B), light branding or no
modification does not push you into Track A.

## 3) Reporting and Payment

- Reporting period: quarterly.
- Report due: within 30 days after quarter end.
- Payment due: within 45 days after quarter end.
- Currency: USD unless otherwise agreed in writing.

Each report must state:
- product/service name,
- track (A or B),
- attributable gross revenue,
- royalty calculation,
- responsible legal entity.

## 4) Audit

Licensor may audit relevant books/records no more than once per year with
30 days written notice.

If underpayment exceeds 5% for a period, licensee must pay:
- underpaid royalties,
- statutory interest where applicable,
- reasonable audit costs.

## 5) Breach and Cure

Failure to report/pay is a material breach.
Licensee has 30 days from notice to cure.
If not cured, commercial rights terminate.

## 6) Separate Enterprise Agreements

Licensor may offer separate written agreements for enterprise, OEM, or custom
terms. A signed separate agreement supersedes these terms for its scope.

## 7) Commercial Fork Grant-Back (Track B)

This section applies only to Track B (Substrate Republication).

If you commercially use, distribute, host, or otherwise monetize a Custom Fork
/ Derivative, you grant Licensor a perpetual, worldwide, non-exclusive,
irrevocable, royalty-free, sublicensable license to use, reproduce, modify,
distribute, publicly perform, and publicly display your Substrate-Grade
Improvements.

Substrate-Grade Improvements means modifications to files under these paths:
- `hardware_native/src/renderer/substrate/**`
- `hardware_native/src/hardware_native/substrate_*.cuh`
- `hardware_native/src/hardware_native/substrate_*.cpp`
- `hardware_native/src/hardware_native/substrate_*.h`
- `hardware_native/src/hardware_native/speech_emission_codec.cuh`
- `hardware_native/include/substrate/**`
- `hardware_native/tests/substrate_*`
- `docs/0x1_handbook_complete.md`
- `docs/substrate/**`
- `docs/0x1_doctrine_proof_matrix.md`

and any equivalent renamed/moved files in the same functional scope.

Functional scope means learning-path logic, substrate laws, kernel/runtime
wiring, stream/route execution surfaces, and core tests/proofs that validate
those changes.

Substrate-Grade Improvements does not include:
- your unrelated proprietary business logic,
- unrelated UI/branding assets,
- unrelated integrations not derived from the substrate core.

On Licensor's written request, you must provide corresponding source code for
Substrate-Grade Improvements that are commercially deployed under Track B.
