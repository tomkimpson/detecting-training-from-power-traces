# Research talk

The source deck is [`slides.md`](slides.md). It is a 21-slide main talk followed by
appendix material, written for a technically literate ML / AI-governance audience.

Render from the repository root with Marp CLI:

```sh
npx --yes @marp-team/marp-cli@4.5.0 --html --allow-local-files talk/slides.md --pdf --output talk/slides.pdf
npx --yes @marp-team/marp-cli@4.5.0 --html --allow-local-files talk/slides.md --output talk/slides.html
```

The deck uses the paper's generated figures directly from `figures/`; no copied figure
assets are maintained under `talk/`.
