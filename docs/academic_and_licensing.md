# Academic use and licensing check

Checked against public primary sources on 16 October 2026. This is an evidence checklist, not legal advice or a declaration signed on behalf of the team.

## CA6129 AI disclosure — course-specific confirmation still required

The supplied assignment slides specify the report, separate contribution PDF and source ZIP. They do not settle permitted AI uses or a declaration format. NTU's public student guidance says permission depends on the course and recommends checking with the instructor. Do not substitute another course's policy or the university's research-publication policy for CA6129 coursework instructions.

Check CA6129's NTULearn syllabus/announcements and submission page for: allowed uses, required declaration, individual versus group declaration, signatures, prompt/output records and upload location. No course-specific AI policy was located in the public sources checked. Do not claim that use was limited to proofreading: the recorded workflow also involved code assistance, experiment orchestration, analysis and report revision. Preserve the actual interaction record privately; do not publish the entire chat or private course material to GitHub.

Source: NTU student guidance on GenAI, "GenAI – is there a right way to use it?" (NTU website).

## CarDD — not an unrestricted public-domain dataset

The official licensing form permits statistical/scientific research subject to prior PIC Lab consent, requires separate authorization for commercial use, restricts redistribution of all or part of the dataset without prior authorization, and requires citing the CarDD paper. Being able to download a copy does not itself document consent.

The team must retain its authorization/license correspondence and confirm publication rights. Dataset images and original annotations are excluded from this code ZIP. The latest remote main removes the old YOLO image examples, and this audit fast-forwards the local checkout to that existing cleanup without restoring those files. They remain recoverable in Git history. Current public main still contains `CarDD project/manifests/cardd_manifest.json`: a 23 MB manifest with 4,000 image records and 8,740 instance annotations. Excluding it from the ZIP does not remove it from GitHub. Confirm redistribution authorization before deciding whether to retain/remove it or handle old history. This audit does not independently delete the manifest or rewrite history.

Sources: official CarDD project page and its official licensing form (CarDD_license.pdf). Required paper: X. Wang, W. Li and Z. Wu, CarDD: A New Dataset for Vision-Based Car Damage Detection, IEEE TITS 24(7), 7202–7214, 2023, DOI 10.1109/TITS.2023.3258480.

## Software and team code

Ultralytics supplies YOLO under AGPL-3.0 or a separate commercial license. Academic use is not a blanket license exemption. Torchvision uses BSD-3-Clause with notice requirements for redistribution. Review the exact versions and any copied third-party portions before publishing a combined project or changing its license. Dependency import is not proof that all project-owned code has already been licensed correctly.

`legacy_maskrcnn.py` is the recovered team-provided trainer snapshot; preserve its provenance and confirm the contributing member's permission for public distribution. The audit adds a notice inventory but does not relicense the team's work or represent that the existing public repository has completed a license review.

See `THIRD_PARTY_NOTICES.md`. Public-release blockers are course-specific disclosure confirmation, CarDD consent/image-redistribution evidence, and the team's license decision.
