/* Re-flow each Plotly graph when its surrounding .plot-card is resized.
   Used by the per-group trajectory HTML files: the `__IDS__` placeholder is
   replaced with the JSON array of graph div ids before embedding. */
(function () {
  var ids = __IDS__;
  function attach() {
    ids.forEach(function (id) {
      var gd = document.getElementById(id);
      if (!gd) return;
      var card = gd.closest(".plot-card");
      if (!card) return;
      var ro = new ResizeObserver(function () {
        if (window.Plotly && Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(gd);
      });
      ro.observe(card);
    });
  }
  if (document.readyState === "complete") attach();
  else window.addEventListener("load", attach);
})();
