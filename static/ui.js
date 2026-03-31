(function () {
  const buttons = document.querySelectorAll("[data-target]");
  const panels = document.querySelectorAll("[data-panel]");

  function activate(target) {
    panels.forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.panel === target);
    });
    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.target === target);
    });
    if (target) {
      history.replaceState(null, "", `#${target}`);
    }
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => activate(button.dataset.target));
  });

  const initial = window.location.hash.replace("#", "");
  if (initial) {
    activate(initial);
  }
})();

