# Group 15 delivery audit — 16 October 2026

## Scope and version

Local checkout was fast-forwarded from stale commit `1ea5a61` to the existing remote `main` commit `b10c37e`. The two upstream commits do not overlap the original uncommitted packaging/test changes or this audit's new files. No force push, remote write or history rewrite was performed. Upstream removal of the superseded YOLO folder was respected; those old files remain recoverable in Git history. The original Colab source under `CarDD project/cardd_code/` is retained in the code package, not the manifests, image results or notebook outputs.

The authoritative report, PPT and contribution Word files were inspected read-only. No manuscript, slide, contribution claim or experiment result was changed by this audit.

## Seven requested checks

| Check | Result | Evidence / remaining action |
|---|---|---|
| Seven experiment families have code | covered locally | shared evaluator, both size retrainings, components, reconstruction selection, threshold policies, AP/F1 bootstrap and two timing cohorts; see experiment map |
| Installation/data/order/output instructions | updated | main README, experiment README, reproduction guide, local-input config and workspace helper |
| Stale experiment status | updated | mock application is distinct from completed model experiments; missing historical YOLO epoch log remains disclosed |
| Tests and clean-directory execution | passed on extracted ZIP | 22 pytest checks passed, one optional PyTorch adapter test skipped; 15 experiment unit tests passed; mock example, 131 bundle checks, five plots and repack all passed; no new GPU training campaign |
| Compact metrics/config/plot entry points | covered | aggregate-only metrics, source hashes and numerical plotting; raw outputs/images/weights excluded; ZIP manifest verifies file bytes and hashes |
| Report/PPT/contribution agreement | needs limited presentation/identity correction | 183 selected report-table numeric checks passed; report/contribution member identifiers agree; PPT issues below |
| AI disclosure and use conditions | partially verified, release confirmation needed | official CarDD terms and software licenses checked; course-specific AI policy, actual dataset authorization and team licensing decision are not established |

## Reproducibility evidence

- 131 compact-bundle arithmetic/protocol checks passed.
- The ZIP was extracted to a new temporary directory and executed with a fresh isolated Python environment, without the original repository on PYTHONPATH. Repacking produced exactly the same source-file hashes. The bundle contains 84 source/config/summary files, not runtime output directories.
- The real exported historical YOLO/retrained Mask predictions were used to rerun 2,000 paired image-bootstrap F1 draws. The point difference `0.01950879029795405` and interval `[-0.008743711628281112, 0.04914708856014973]` exactly reproduce the retained result.
- Numeric report checks cover Tables 5, 6 counts, 8, 10, 11, 12 mean/median and 13. They are not a claim that every word or figure was independently validated.
- The compact figure entry recreates source-driven evidence for Figures 1, 2, 4, 5 and 6. Photographic Figure 3 needs authorized images and exported predictions outside the package.
- `SUBMISSION_MANIFEST.json` records actual URL presence. Official documentation/license/package-index URLs are not cloud-hosted substitute code, but a literal ban on every URL needs instructor clarification. The builder does not conceal this distinction.

## PPT findings — not edited

1. Slide 2 says “four findings” while the deck contains three finding slides.
2. Slide 6 says “AP is a tie”. The reported interval spans zero; this supports no clearly resolved difference, not a formal equivalence conclusion. Use “AP is close; the interval spans zero”.
3. Slide 7 cites 76 versus 84 ms without the low score floor and RGB-to-RLE boundary. Add those qualifiers or remove the speed comparison. Its “settings move AP more than model choice” also combines different comparisons; limit the statement to the tested reconstruction setting.
4. Slide 8 says failure is “not at” proposals, although 3/70 cracks lack proposal coverage at IoU 0.5. “Losses occur mainly after proposals” matches the diagnosis.
5. Native chart 4 has YOLO tire AP `0.815`; the common all-area source is `0.813765...` (`0.814` at three decimals). Rebuild that chart from the compact common-evaluation values rather than mixing area-specific approximations.
6. PPT labels Q2=settings and Q3=crack, while report RQ2=class errors and RQ3=resolution. The report explains the different presentation order; consider using “Finding 2/3” rather than implying identical RQ numbering.
7. Member B is identified by nickname in the report/contribution but by a campus email on the PPT cover. Confirm identity and use one agreed display convention. This audit does not invent a legal/student-card name or silently rename an author.
8. Slides also retain small stray text characters, including a cover “z” and a “v” near the last finding; check visually before final recording.

## Release confirmations still needed

- CA6129-specific GenAI permission, declaration format, paper-trail requirement and submission location must come from that course's NTULearn/instructor. Do not borrow another course's policy or declare “proofreading only” for a broader AI-assisted workflow.
- CarDD's official form requires prior research consent and restricts redistribution. Current public `main` includes `CarDD project/manifests/cardd_manifest.json`, with 4,000 image records and 8,740 annotations. It is not in this ZIP. Decide its public availability only after checking authorization; deletion or historical cleanup requires a separate scoped decision.
- GitHub currently reports no repository-wide license. The third-party notice inventory is not a license grant. Confirm team permission and a compatible license before treating the project as a licensed public release.
- No remote commit, push or PR is performed by this audit. Local staging is not the same as publication. The final code ZIP must correspond to the version eventually committed.

## Validation commands

```sh
python -m pytest -q
python -m unittest discover -s experiments -p 'test_*.py' -v
python reporting/verify_metrics.py
python reporting/plot_metrics.py --output outputs/report_figures
python scripts/build_submission_zip.py --group-id 15
```

The private audit directory contains the exact source-file hashes, read-only document extraction, raw test logs and clean-directory acceptance record. None of those private document extracts, student identifiers, datasets, model weights or runtime predictions is included in the code archive.
