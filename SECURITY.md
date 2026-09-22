# Security policy

Version 0.1.x is the initial supported development line. Security fixes are
applied to the latest development version; there is no promised response SLA.

Do not post exploit details, private images or credentials in a public issue.
When this repository is hosted, use private vulnerability reporting if enabled,
or an available private maintainer contact. A dedicated security address and
private-reporting URL have not yet been configured for this local repository.
The repository owner should configure them before a public release.

A useful report includes affected version, environment, minimal reproducer,
impact and any suggested mitigation. Coordinate public disclosure after a fix.

This package executes GPU code in the calling process. It is not a sandbox for
untrusted Python, Torch tensors or extensions. CLI image decoding relies on
Pillow; keep it updated and follow your application's input size/resource limits.
Do not load untrusted comparator modules through the developer benchmark tools.
