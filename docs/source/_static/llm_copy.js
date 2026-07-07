// Adds a "Copy for LLM" button to each docs page. It copies the page's
// readable text (title + URL + body) to the clipboard, formatted for pasting
// into an AI assistant. Pairs with the site-level /llms.txt index.
(function () {
  function getMain() {
    return (
      document.querySelector('[role="main"]') ||
      document.querySelector("div.body") ||
      document.querySelector("div.document")
    );
  }

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
    btn.textContent = "Copy for LLM";
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pageContent()).then(
        function () {
          btn.textContent = "Copied!";
          setTimeout(function () { btn.textContent = "Copy for LLM"; }, 1500);
        },
        function () {
          btn.textContent = "Copy failed";
          setTimeout(function () { btn.textContent = "Copy for LLM"; }, 1500);
        }
      );
    });
    return btn;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var main = getMain();
    if (main && navigator.clipboard) {
      main.insertBefore(makeButton(), main.firstChild);
    }
  });
})();
