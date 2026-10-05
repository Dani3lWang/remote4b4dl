"""A restrained results-page theme shared by the Gradio entry point."""

APP_CSS = """
:root {
  --b4-text: #242424;
  --b4-muted: #646464;
  --b4-line: #d9d9d4;
  --b4-accent: #385a7c;
}
body { background: #f3f2ef !important; }
.gradio-container {
  max-width: 1480px !important;
  min-height: 100vh;
  padding: 28px 40px 40px !important;
  background: #ffffff !important;
  color: var(--b4-text) !important;
  font-family: Arial, "Noto Sans CJK SC", sans-serif !important;
}
#b4-hero { border-bottom: 2px solid #242424; padding: 0 0 20px; margin-bottom: 18px; }
.b4-kicker { color: var(--b4-muted); font-size: 12px; letter-spacing: .13em; }
.b4-title {
  color: var(--b4-text) !important; margin: 10px 0 8px;
  font-family: "Times New Roman", "Noto Serif CJK SC", serif;
  font-size: clamp(28px, 3vw, 40px); line-height: 1.2; font-weight: 600;
}
.b4-subtitle { color: var(--b4-muted); font-size: 14px; line-height: 1.7; }
.b4-reference { display: inline-block; margin-top: 10px; color: var(--b4-accent); font-size: 12px; }
.effect-shell { padding: 0 !important; border: 0 !important; background: #ffffff !important; }
.gradio-container .styler, .gradio-container .gr-group { background: #ffffff !important; }
.effect-heading {
  margin: 0 0 10px; font-family: "Times New Roman", "Noto Serif CJK SC", serif;
  color: var(--b4-text); font-size: 23px; font-weight: 600;
}
.metric-rail { overflow-x: auto; margin: 4px 0 14px; }
.results-table { width: 100%; min-width: 860px; border-collapse: collapse; font-size: 14px; }
.results-table caption { text-align: left; color: var(--b4-muted); font-size: 12px; padding: 0 0 8px; }
.results-table thead { border-top: 2px solid #242424; border-bottom: 1px solid #777; }
.results-table tbody { border-bottom: 2px solid #242424; }
.results-table th, .results-table td { padding: 11px 14px; text-align: right; white-space: nowrap; }
.results-table th:first-child { text-align: left; }
.results-table thead th { font-weight: 500; font-family: "Times New Roman", serif; }
.results-table tbody th { font-weight: 500; }
.results-table td { font-variant-numeric: tabular-nums; }
.gradio-container .results-table, .gradio-container .results-table tr, .gradio-container .results-table th,
.gradio-container .results-table td { border: 0 !important; }
.results-table .current-run { background: #f7f8fa; }
.results-table .current-run td { font-weight: 600; }
.results-table .paper-reference { color: var(--b4-muted); }
.results-table .is-na { color: #767676; font-weight: 400; }
.effect-warning {
  margin: 4px 0 14px; padding: 8px 11px; border-left: 2px solid #9b7e42;
  background: #faf8f2; color: #6b5730; font-size: 12px; line-height: 1.7;
}
.results-note { color: var(--b4-muted); font-size: 13px; line-height: 1.7; margin: 0 0 12px; }
.figure-caption {
  margin: 5px 0 18px; border-top: 1px solid var(--b4-line); padding: 10px 2px 0;
  color: var(--b4-muted); font-size: 13px; line-height: 1.7;
}
.figure-caption strong { color: var(--b4-text); font-family: "Times New Roman", "Noto Serif CJK SC", serif; }
.gradio-container .tab-container { border-bottom: 1px solid var(--b4-line) !important; gap: 8px; }
.gradio-container button[role="tab"] { border-radius: 0 !important; padding: 12px 16px !important; }
.gradio-container button[role="tab"][aria-selected="true"] {
  color: var(--b4-text) !important; border-bottom: 2px solid var(--b4-text) !important;
  background: #ffffff !important; font-weight: 600;
}
.gradio-container button[role="tab"][aria-selected="true"]::after { background: #242424 !important; }
button.primary {
  background: #385a7c !important; color: #ffffff !important;
  border: 1px solid #385a7c !important; box-shadow: none !important;
}
button.secondary { background: #ffffff !important; border-color: #b9b9b4 !important; box-shadow: none !important; }
button:focus-visible, a:focus-visible { outline: 2px solid #385a7c !important; outline-offset: 3px; }
textarea, input { font-family: Arial, "Noto Sans CJK SC", sans-serif !important; }
.diagnosis-strip { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 12px; padding: 10px 0; border-bottom: 1px solid var(--b4-line); }
.diagnosis-strip span { color: var(--b4-muted); font-size: 13px; }
.diagnosis-strip span:first-child { color: var(--b4-text); }
#effect-sample-panel { border-left: 1px solid var(--b4-line); padding-left: 20px; }
#paper-case-builder { padding: 14px !important; background: #fafaf8 !important; border: 1px solid var(--b4-line); }
#paper-case-preview { background: #ffffff !important; border: 1px solid var(--b4-line); padding: 8px !important; }
#paper-preview-column { padding: 0 !important; }
.evidence-key { display: flex; flex-wrap: wrap; gap: 18px; margin: 10px 0 16px; font-size: 12px; color: var(--b4-muted); }
.evidence-key i { display: inline-block; width: 14px; height: 10px; border: 1px solid #777; margin-right: 6px; }
.evidence-key .front { background: #ffe84a; }
.evidence-key .rear { background: #70f08b; }
.evidence-key .target { background: #b770df; }
.evidence-key .error { color: #c51e28; }
#b4-controls, #b4-viewer, #b4-chat { border: 1px solid var(--b4-line) !important; background: #ffffff !important; padding: 14px !important; }
#b4-frame-summary, #b4-status { color: var(--b4-muted); font-size: 13px; line-height: 1.7; }
@media (min-width: 1100px) { #paper-preview-column { position: sticky; top: 16px; align-self: flex-start; } }
@media (max-width: 800px) {
  .gradio-container { padding: 18px 16px 28px !important; }
  .gradio-container button[role="tab"] { padding: 10px 8px !important; font-size: 13px; }
  #effect-sample-panel { border-left: 0; padding-left: 0; }
}
"""

HERO_HTML = """
<header id="b4-hero">
  <div class="b4-kicker">B4DL · EXPERIMENTAL RESULTS</div>
  <h1 class="b4-title">4D LiDAR 模型实验结果</h1>
  <div class="b4-subtitle">定量评测、定性案例与训练过程，按论文结果页组织展示。</div>
  <a class="b4-reference" href="https://arxiv.org/abs/2508.05269" target="_blank" rel="noopener noreferrer">
    论文参考：Table 3 · Figure 5 · Figure 8 ↗
  </a>
</header>
"""

# Hidden tabs can leave Plotly at its initial 700 px width. Resize the plot when
# its container becomes visible or changes size, without changing trace data.
PLOT_RESIZE_JS = """() => {
  const root = document.querySelector('.gradio-container') || document.body;
  if (!root || root.__b4dlResizeInstalled) return [];
  root.__b4dlResizeInstalled = true;
  const observed = new WeakSet();
  let scheduled = false;
  const resize = (plot) => {
    if (!plot._fullLayout || plot.clientWidth < 1 || !window.Plotly?.Plots) return;
    if (Math.abs(plot._fullLayout.width - plot.clientWidth) < 2) return;
    window.Plotly.Plots.resize(plot).catch(() => {});
  };
  const observer = new ResizeObserver(entries => entries.forEach(entry => resize(entry.target)));
  const discover = () => {
    scheduled = false;
    root.querySelectorAll('.js-plotly-plot').forEach(plot => {
      if (!observed.has(plot)) { observed.add(plot); observer.observe(plot); }
      resize(plot);
    });
  };
  const schedule = () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(discover); }
  };
  new MutationObserver(schedule).observe(root, {childList: true, subtree: true});
  discover();
  return [];
}"""


def create_paper_theme():
    """Apply the same readable paper colors in light and system dark modes."""
    import gradio as gr

    theme = gr.themes.Base(primary_hue="blue", secondary_hue="gray", neutral_hue="gray",
                           font=["Arial", "Noto Sans CJK SC", "sans-serif"],
                           font_mono=["Consolas", "monospace"])
    tokens = {
        "body_background_fill": "#ffffff", "body_text_color": "#242424",
        "body_text_color_subdued": "#646464", "background_fill_primary": "#ffffff",
        "background_fill_secondary": "#f7f7f4", "border_color_primary": "#d9d9d4",
        "border_color_accent": "#385a7c", "block_background_fill": "#ffffff",
        "block_border_color": "#d9d9d4", "block_shadow": "none", "block_radius": "2px",
        "block_label_background_fill": "#ffffff", "block_label_text_color": "#535353",
        "block_title_text_color": "#242424", "input_background_fill": "#ffffff",
        "input_border_color": "#ccccca", "input_border_width": "1px",
        "input_border_color_focus": "#385a7c", "input_shadow": "none",
        "input_shadow_focus": "none", "input_radius": "2px",
        "button_primary_background_fill": "#385a7c", "button_primary_background_fill_hover": "#294866",
        "button_primary_text_color": "#ffffff", "button_primary_border_color": "#385a7c",
        "button_secondary_background_fill": "#ffffff", "button_secondary_text_color": "#242424",
        "button_secondary_border_color": "#b9b9b4",
    }
    theme = theme.set(**tokens)
    variables = theme.to_dict()["theme"]
    # Files, inline code, tables and secondary controls also have dark tokens.
    return theme.set(**{key: variables[key[:-5]] for key in variables
                        if key.endswith("_dark") and key[:-5] in variables})
