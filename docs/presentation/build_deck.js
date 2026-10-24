// Builds docs/presentation/decide.pptx (plain, 7 slides, speaker notes).
// Run: NODE_PATH=<dir with pptxgenjs> node build_deck.js <skill apply_theme.js path>
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const THEME = {
  name: "Decide Plain",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2933", lt1: "FFFFFF", dk2: "36454F", lt2: "F2F4F5",
    accent1: "0F766E", accent2: "B45309", accent3: "64748B", accent4: "94A3B8",
    accent5: "CCFBF1", accent6: "FEF3C7", hlink: "0F766E", folHlink: "36454F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";              // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Amal Verma";
pres.title = "Decide";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Title Dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.6, w: 8.8, h: 1.0, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 2.7, w: 8.8, h: 0.9, fontSize: 20, color: C.background2, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "byline", type: "body", x: 0.6, y: 4.6, w: 8.8, h: 0.5, fontSize: 14, color: C.accent4 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.4, 0.5, 0.4, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.7, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3 },
});

const box = (slide, name, x, y, w, h, fill, opts = {}) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: name, x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, ...opts });

// 1. Title -----------------------------------------------------------------
pres.addSection({ title: "Intro" });
let s = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Intro" });
s.addText("Decide", { placeholder: "title" });
s.addText("One small model, three typed decisions, about 10 ms on a CPU", { placeholder: "subtitle" });
s.addText("Amal Verma  ·  Applied ML track", { placeholder: "byline" });
s.addNotes("Decide answers three small questions about a support ticket with one 23 M-parameter model on a CPU, in about 10 ms, with honest confidence and the option to say: let a human decide.");

// 2. Problem -----------------------------------------------------------------
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
s.addText("Three questions per support ticket", { placeholder: "title" });
const cards = [
  ["route", "Which queue?", "billing · technical · account · shipping · general", "choice + 5 probabilities"],
  ["approve", "Auto-approve?", "a wrong approval costs 3x a wrong hold", "yes / no + probability"],
  ["grounded", "Draft backed by the article?", "compare the reply with the KB article", "yes / no + probability"],
];
cards.forEach(([k, q, d, a], i) => {
  const x = 0.5 + i * 3.05;
  box(s, `card-${k}`, x, 1.25, 2.85, 2.35, C.background2);
  s.addText(k, { isTextBox: true, objectName: `card-${k}-key`, x: x + 0.2, y: 1.4, w: 2.45, h: 0.4, fontSize: 14, bold: true, color: C.accent1, margin: 0 });
  s.addText(q, { isTextBox: true, objectName: `card-${k}-q`, x: x + 0.2, y: 1.8, w: 2.45, h: 0.6, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "top" });
  s.addText(d, { isTextBox: true, objectName: `card-${k}-d`, x: x + 0.2, y: 2.45, w: 2.45, h: 0.6, fontSize: 14, color: C.text2, margin: 0, valign: "top" });
  s.addText(a, { isTextBox: true, objectName: `card-${k}-a`, x: x + 0.2, y: 3.1, w: 2.45, h: 0.35, fontSize: 12, italic: true, color: C.accent3, margin: 0 });
});
const rules = ["one shared model", "CPU only, no network", "p95 under 25 ms for full marks", "calibrated", "may abstain (answer 80%+)"];
rules.forEach((r, i) => {
  const x = 0.5 + i * 1.83;
  box(s, `rule-${i}`, x, 4.0, 1.7, 0.75, C.background1, { line: { color: C.accent4, width: 1 } });
  s.addText(r, { isTextBox: true, objectName: `rule-${i}-text`, x: x + 0.08, y: 4.0, w: 1.54, h: 0.75, fontSize: 12, color: C.text1, align: "center", valign: "middle", margin: 0 });
});
s.addNotes("Today products call an LLM for these: slow, costly, no usable confidence. The brief asks for one small model. Each call names one decision. Approve is asymmetric: a wrong approval costs three times a wrong hold, and part of the rule is hidden account history.");

// 3. How it works ------------------------------------------------------------
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Approach" });
s.addText("One shared model, a tiny head per question", { placeholder: "title" });
const steps = [
  ["ticket +\nquestion", C.background2, C.text1],
  ["frame input\nper question", C.background2, C.text1],
  ["MiniLM\nencoder", C.accent1, C.background1],
  ["mixture of\n4 experts", C.accent1, C.background1],
  ["tiny head\nper question", C.accent2, C.background1],
  ["calibrate +\nabstain rule", C.background2, C.text1],
];
steps.forEach(([t, fill, fg], i) => {
  const x = 0.5 + i * 1.53;
  box(s, `step-${i}`, x, 1.35, 1.33, 1.0, fill);
  s.addText(t, { isTextBox: true, objectName: `step-${i}-text`, x, y: 1.35, w: 1.33, h: 1.0, fontSize: 13, bold: true, color: fg, align: "center", valign: "middle", margin: 0 });
  if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_TRIANGLE, { objectName: `arrow-${i}`, x: x + 1.37, y: 1.77, w: 0.12, h: 0.16, rotate: 90, fill: { color: C.accent3 }, line: { color: C.accent3 } });
});
s.addText([{ text: "Teal = shared by all three questions.  ", options: { bold: true, color: C.accent1 } },
           { text: "Amber = the only per-question part.", options: { bold: true, color: C.accent2 } }],
          { isTextBox: true, objectName: "legend", x: 0.5, y: 2.5, w: 9.0, h: 0.35, fontSize: 13, margin: 0 });
const stats3 = [["23 M", "parameters, 18x smaller than Laya"], ["99.99%", "of weights shared"], ["24 MB", "int8 ONNX on CPU"]];
stats3.forEach(([big, small], i) => {
  const x = 0.5 + i * 3.05;
  s.addText(big, { isTextBox: true, objectName: `stat3-${i}`, x, y: 3.15, w: 2.85, h: 0.8, fontSize: 40, bold: true, color: C.accent1, margin: 0 });
  s.addText(small, { isTextBox: true, objectName: `stat3-${i}-label`, x, y: 3.95, w: 2.85, h: 0.45, fontSize: 14, color: C.text2, margin: 0 });
});
s.addNotes("Each question is framed for what it compares: route reads the ticket, approve also gets tier and amount features, grounded reads article and draft side by side. Everything in teal is shared; a fourth decision is one more tiny head. Trained on a free Kaggle GPU in about 3 minutes, served as int8 ONNX.");

// 4. Results ---------------------------------------------------------------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
s.addText("Results with the judges' own scoring code", { placeholder: "title" });
const stats4 = [["83.75", "dev score / 100"], ["82.6 ± 1.3", "4-fold cross-validation"], ["10.6 ms", "mean p95 latency"], ["0.016", "calibration error (ECE)"]];
stats4.forEach(([big, small], i) => {
  const y = 1.2 + i * 0.95;
  s.addText(big, { isTextBox: true, objectName: `stat4-${i}`, x: 0.5, y, w: 2.9, h: 0.6, fontSize: 30, bold: true, color: C.accent1, margin: 0 });
  s.addText(small, { isTextBox: true, objectName: `stat4-${i}-label`, x: 0.5, y: y + 0.55, w: 2.9, h: 0.3, fontSize: 12, color: C.text2, margin: 0 });
});
s.addChart(pres.charts.BAR, [{ name: "Score (0 to 1)", labels: ["Quality (40%)", "Calibration (25%)", "Latency (20%)", "Abstention (15%)"], values: [0.849, 0.856, 1.0, 0.558] }], {
  objectName: "chart-axes", x: 3.7, y: 1.15, w: 5.8, h: 3.8, barDir: "bar",
  chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  dataLabelColor: HEX.dk1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.15,
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Score per axis (served model, dev)", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
});
s.addNotes("Every gate passes: 4,500 of 4,500 responses valid. Latency is far under the 25 ms line. Calibration is very honest. Abstention is the weakest axis because some labels are randomly wrong. Expect about 82.5 to 84 on the hidden test.");

// 5. Refusing --------------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
s.addText("Knowing when not to answer", { placeholder: "title" });
s.addText([
  { text: "Approve only above 75%", options: { bold: true, color: C.text1, breakLine: true } },
  { text: "A wrong approval costs 3x a wrong hold. Approving pays off only when 3(1 - p) < p, that is, when p > 0.75.", options: { color: C.text2, breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Refuse the risky middle", options: { bold: true, color: C.text1, breakLine: true } },
  { text: "Approve abstains when 0.15 < p < 0.79: lopsided, because confident-looking approvals are the costly mistakes.", options: { color: C.text2 } },
], { isTextBox: true, objectName: "refuse-text", x: 0.5, y: 1.2, w: 4.4, h: 3.2, fontSize: 15, valign: "top", margin: 0, paraSpaceAfter: 4 });
box(s, "example-card", 5.3, 1.2, 4.2, 3.4, C.background2);
s.addText("Live example", { isTextBox: true, objectName: "example-head", x: 5.55, y: 1.35, w: 3.7, h: 0.35, fontSize: 13, bold: true, color: C.accent1, margin: 0 });
s.addText("Pro customer, $112 (near the ~$120 pro limit), ticket not about money", { isTextBox: true, objectName: "example-body", x: 5.55, y: 1.75, w: 3.7, h: 0.75, fontSize: 14, color: C.text1, margin: 0, valign: "top" });
s.addText("p_true = 0.31", { isTextBox: true, objectName: "example-p", x: 5.55, y: 2.65, w: 3.7, h: 0.6, fontSize: 30, bold: true, color: C.text1, margin: 0 });
s.addText("abstain: true  →  a human decides", { isTextBox: true, objectName: "example-out", x: 5.55, y: 3.3, w: 3.7, h: 0.45, fontSize: 16, bold: true, color: C.accent2, margin: 0 });
s.addText("(the labelled answer was: hold)", { isTextBox: true, objectName: "example-note", x: 5.55, y: 3.8, w: 3.7, h: 0.35, fontSize: 12, italic: true, color: C.accent3, margin: 0 });
s.addNotes("Demo: send the $112 Pro ticket. The model says 0.31 and abstains instead of guessing. Each decision refuses roughly its least-confident 15%, keeping coverage above the 80% target.");

// 6. Trade-off + what we tried ------------------------------------------------
pres.addSection({ title: "Trade-offs" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Trade-offs" });
s.addText("What we tried, and the trade-off we made", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Dev score", labels: ["Pure NN from scratch", "Classical baseline", "Ettin-32m (2025)", "MiniLM + MoE (served)", "Bigger H768 + MoE"], values: [77.7, 78.45, 83.14, 83.75, 84.52] }], {
  objectName: "chart-tried", x: 0.4, y: 1.1, w: 5.4, h: 3.9, barDir: "bar",
  chartColors: [HEX.accent4], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
  dataLabelColor: HEX.dk1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valAxisMinVal: 70, valAxisMaxVal: 88,
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Dev score / 100", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
});
box(s, "tradeoff-card", 6.1, 1.2, 3.4, 3.6, C.background2);
s.addText([
  { text: "The trade-off", options: { bold: true, color: C.accent1, breakLine: true } },
  { text: "The bigger H768 scored 84.5 once, but 83.1 when retrained: its real edge is about +0.2, inside the noise. It runs 2x slower (21\u00A0ms p95), and graph fusion did not speed it up.", options: { color: C.text1, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "We ship the small model: latency is 20% of the score.", options: { bold: true, color: C.text1 } },
], { isTextBox: true, objectName: "tradeoff-text", x: 6.3, y: 1.35, w: 3.0, h: 3.3, fontSize: 14, valign: "top", margin: 0 });
s.addNotes("The one real trade-off: the bigger H768 model looked better in one run, but retraining showed most of that was luck, and it is twice as slow. Latency is 20% of the score, so we ship the small model.");

// 7. Everything else tested ------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Trade-offs" });
s.addText("Everything else we tested", { placeholder: "title" });
const tested = [["MC dropout (Bayesian)", 0.09], ["Ettin-17m (2025) + calibration", 0.08], ["Entropy regularisation", 0.04],
  ["Augmentation 60%", 0.04], ["Distillation from H768", 0.02], ["Laplace (Bayesian)", -0.23], ["Canonical durations", -0.40],
  ["Focal loss", -0.46], ["Ettin-32m (2025)", -0.47], ["Label smoothing", -0.61], ["Shared trunk, no experts", -0.99],
  ["Label cleaning", -1.02], ["Augmentation 30%", -1.22]];
s.addChart(pres.charts.BAR, [{ name: "Change vs served", labels: tested.map(t => t[0]), values: tested.map(t => t[1]) }], {
  objectName: "chart-tested", x: 0.4, y: 1.05, w: 5.9, h: 4.2, barDir: "bar",
  chartColors: [HEX.accent3], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00",
  dataLabelColor: HEX.dk1, dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  catAxisLabelPos: "low", valAxisHidden: true, valAxisMinVal: -1.6, valAxisMaxVal: 0.5,
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Score change vs served model (dev, same seeds)", titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
});
box(s, "tested-card", 6.6, 1.2, 2.9, 3.9, C.background2);
s.addText([
  { text: "Nothing beats it", options: { bold: true, color: C.accent1, breakLine: true } },
  { text: "Every gain is within the seed noise (about ±0.5).", options: { color: C.text1, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "A from-scratch network scored 77.7 (-5.9): the pretrained encoder is what matters.", options: { color: C.text1, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Cleaning or augmenting labels makes the model overconfident on the noisy test.", options: { color: C.text1 } },
], { isTextBox: true, objectName: "tested-text", x: 6.8, y: 1.35, w: 2.5, h: 3.6, fontSize: 13, valign: "top", margin: 0 });
s.addNotes("We tested newer encoders, Bayesian methods, loss functions, preprocessing, label cleaning and data augmentation, each paired on the same seeds. Nothing beats the served model by more than the noise, because the remaining errors are random labels.");

// 7. Conclusion ---------------------------------------------------------------
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Close" });
s.addText("At the data's ceiling", { placeholder: "title" });
s.addText("About 6% of route labels and 15% of approvals are randomly wrong; every method we tested hits the same wall. Honest estimate: 82.5 to 84 on the hidden test.", { placeholder: "subtitle" });
s.addText("Adding a fourth decision = one tiny head, same speed", { placeholder: "byline" });
s.addNotes("Close: the remaining errors are label noise that no model can learn, and our probabilities say so honestly. The design scales: a new decision is one more head on the same shared model.");

(async () => {
  const out = path.join(__dirname, "decide.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
