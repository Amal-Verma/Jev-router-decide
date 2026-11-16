# Interface design and behavior

## Direction

The workbench uses an editorial style: warm paper, ink-colored text, restrained teal and coral accents, and large serif headings paired with quiet sans-serif labels. A single request on the left flows into three numbered decisions on the right. The confidence rings and routing bars are the visual anchor. The design feasibility score is 14: impact 4, context fit 5, feasibility 5, performance 5, consistency risk 5.

Local Noto Serif supplies expressive headings; Noto Sans keeps forms and measurements legible. The palette and spacing live in CSS variables and a consistent small spacing rhythm. Bundled font files and their SIL Open Font License are in `static/fonts/`. No remote fonts, libraries, images or analytics are requested. There are no emojis; small action icons use inline SVG or CSS geometry.

## Interaction

`static/index.html` defines the form and accessible result regions. `static/app.js` owns preset selection, input revisions, requests, response validation and result rendering. `static/styles.css` defines the visual system, layout, focus styles and motion.

The Refund, Account access and Mixed request presets populate editable input fields. They are demonstration inputs only: predictions always come from the trained service. Running the form makes three sequential `POST /v1/decide` calls using the same state, one per decision. Results are committed together only when all three responses validate. Each request has a unique identifier whose echo is checked.

Routing shows all five probabilities. Binary cards distinguish the probability of a yes answer from confidence in the displayed answer. Abstained binary results prominently say human review is needed. An approval hold without abstention is presented as a model recommendation to hold. No refund or other customer action is executed by the interface.

The form disables repeated submissions while running, shows loading state and retains previous results on failure. Edited inputs mark existing predictions stale; edits during a request are also detected. Fetch timeouts and network/HTTP failures produce visible errors. The health badge reflects `/health`; stored dev score comes from the selected model’s `/v1/metrics`, and timing comes from the latest actual model responses. Missing evaluation reports leave the score blank. The raw response disclosure supports copy-to-clipboard with a selection fallback.

## Layout, accessibility and motion

The desktop layout has two columns with asymmetric content inside them. Below 780 CSS pixels it becomes one column; below 420 pixels spacing and controls compact further. Headings and labels have clear hierarchy; inputs have explicit labels; results use a polite live region, loading uses `aria-busy`, errors use an alert, and keyboard focus is visible. Ctrl/Cmd+Enter runs the form. Buttons and native disclosures remain keyboard operable. Text is rendered through safe DOM text APIs rather than interpolated HTML.

Result cards enter in a short stagger, probability bars ease to their new values, and hover/focus transitions are restrained. `prefers-reduced-motion` disables animation and transitions.

## Verification

The original workbench was checked in a live browser for form submission, uncertainty display, JSON inspection and service recovery. The consolidated interface is separately checked against the retained ONNX ensemble; current checks are recorded in `CONSOLIDATION.md`. The backend contract tests separately verify that the interface and static assets are served. Responsive breakpoints are implemented; physical-device behavior is not certified.
