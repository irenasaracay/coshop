// Small interactions: results figure tabs.
(function () {
  const tabs = document.querySelectorAll("#result-tabs .tab");
  const panels = document.querySelectorAll("#results .tabpanel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.panel;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) => p.classList.toggle("active", p.dataset.panel === key));
    });
  });
})();
