(() => {
  const result = window.__RESULT__;
  if (!result) return;

  const ctx = document.getElementById("predictionChart");
  const explanationCtx = document.getElementById("explanationChart");

  const approval = Number(result.approval_probability ?? 0);
  const risk = Number(result.risk_score ?? 0) / 100;
  const anomaly = Object.values(result.anomaly_scores || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const anomalyAvg = anomaly ? anomaly / Object.keys(result.anomaly_scores || {}).length : 0;

  if (ctx) {
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Approval", "Risk", "Anomaly"],
        datasets: [
          {
            data: [
              Math.max(approval, 0.01),
              Math.max(risk, 0.01),
              Math.max(anomalyAvg, 0.01),
            ],
            backgroundColor: ["#2dd4bf", "#7c8cff", "#ff7a7a"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "#eef4ff",
              usePointStyle: true,
              padding: 18,
            },
          },
        },
        cutout: "68%",
      },
    });
  }

  if (explanationCtx && Array.isArray(result.top_features) && result.top_features.length) {
    const labels = result.top_features.map((item) => item.feature);
    const values = result.top_features.map((item) => Math.abs(Number(item.impact ?? 0)));
    const colors = result.top_features.map((item) => Number(item.impact ?? 0) >= 0 ? "#2dd4bf" : "#ff7a7a");

    new Chart(explanationCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Feature impact",
            data: values,
            backgroundColor: colors,
            borderWidth: 0,
            borderRadius: 10,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#a8b5cc",
            },
            grid: {
              color: "rgba(255,255,255,0.08)",
            },
          },
          y: {
            ticks: {
              color: "#eef4ff",
            },
            grid: {
              display: false,
            },
          },
        },
      },
    });
  }
})();
