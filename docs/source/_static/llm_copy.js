// Adds a "Copy for LLM" button to each docs page. It copies the page's
// readable text (title + URL + body) to the clipboard, formatted for pasting
// into an AI assistant. Pairs with the site-level /llms.txt index.
(function () {
  // Where the button goes. Prefer the article container so the button sits
  // with the prose; fall back through older theme structures.
  function getHost() {
    return (
      document.querySelector(".bd-article") ||
      document.querySelector('[role="main"]') ||
      document.querySelector("div.body") ||
      document.querySelector("div.document")
    );
  }

  // Where the text comes from. Always the main region, never just the article,
  // so the copied content is the same regardless of theme.
  function getMain() {
    return (
      document.querySelector('[role="main"]') ||
      document.querySelector("div.body") ||
      document.querySelector("div.document")
    );
  }

  var CLIPBOARD_SVG =
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"' +
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="5" y="2.5" width="6" height="2.5" rx="0.75"/>' +
    '<path d="M11 4h1.5A1.5 1.5 0 0 1 14 5.5v7A1.5 1.5 0 0 1 12.5 14h-9A1.5 1.5 0 0 1 2 12.5v-7A1.5 1.5 0 0 1 3.5 4H5"/>' +
    "</svg>";

  function pageContent() {
    var main = getMain();
    var title = (document.querySelector("h1") || {}).innerText || document.title || "";
    var body = main ? main.innerText : "";
    return "# " + title.trim() + "\n\nSource: " + window.location.href + "\n\n" + body.trim();
  }

  function makeButton() {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "llm-copy-btn";
    // The full purpose lives in the tooltip and the accessible name; the
    // visible label stays short so the button does not dominate the page.
    btn.title = "Copy this page as text, formatted for pasting into an AI assistant";
    btn.setAttribute("aria-label", "Copy page for LLM");

    var label = document.createElement("span");
    label.textContent = "Copy";
    btn.innerHTML = CLIPBOARD_SVG;
    btn.appendChild(label);

    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pageContent()).then(
        function () {
          label.textContent = "Copied";
          setTimeout(function () { label.textContent = "Copy"; }, 1500);
        },
        function () {
          label.textContent = "Failed";
          setTimeout(function () { label.textContent = "Copy"; }, 1500);
        }
      );
    });
    return btn;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var host = getHost();
    if (host && navigator.clipboard) {
      host.insertBefore(makeButton(), host.firstChild);
    }
  });
})();
