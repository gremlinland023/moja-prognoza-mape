# moja-prognoza-mape

Gotove mape vremena za aplikaciju **Moja prognoza** (temperatura, kiša i sneg, 48 h unapred).

- Podaci: model ICON-EU, Deutscher Wetterdienst (DWD), otvoreni podaci — © DWD, izvor: opendata.dwd.de
- Robot (`.github/workflows/mape.yml`) na svaka 3 sata pravi nove slike na granu `mape`.
- `index.json` — vreme proračuna, granice slike, vremena i temperature za gradove.
