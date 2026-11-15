// Traffic Pattern Discovery - Interactive Dashboard Scripts

function formatHour(val) {
  const h = parseFloat(val);
  const hourInt = Math.floor(h);
  const min = Math.round((h - hourInt) * 60);
  const period = hourInt >= 12 ? "PM" : "AM";
  const displayH = hourInt === 0 ? 12 : (hourInt > 12 ? hourInt - 12 : hourInt);
  const minStr = min < 10 ? "0" + min : min;
  return `${displayH}:${minStr} ${period} (${h.toFixed(2)}h)`;
}

function updateHourDisplay(val) {
  const el = document.getElementById("hourVal");
  if (el) el.innerText = formatHour(val);
}

function updateTotals() {
  const cars = parseInt(document.getElementById("carsInput")?.value || 0);
  const bikes = parseInt(document.getElementById("bikesInput")?.value || 0);
  const buses = parseInt(document.getElementById("busesInput")?.value || 0);
  const trucks = parseInt(document.getElementById("trucksInput")?.value || 0);
  
  const total = cars + bikes + buses + trucks;
  const truckRatio = total > 0 ? ((trucks / total) * 100).toFixed(1) : "0.0";
  
  const totalDisp = document.getElementById("totalDisplay");
  if (totalDisp) totalDisp.innerText = total;
  
  const truckDisp = document.getElementById("truckRatioDisplay");
  if (truckDisp) truckDisp.innerText = truckRatio + "%";
}

function loadPreset(type) {
  if (type === "morning_rush") {
    document.getElementById("hourInput").value = 8.5;
    updateHourDisplay(8.5);
    document.getElementById("daySelect").value = "Monday";
    document.getElementById("carsInput").value = 95;
    document.getElementById("bikesInput").value = 30;
    document.getElementById("busesInput").value = 15;
    document.getElementById("trucksInput").value = 4;
  } else if (type === "night_freight") {
    document.getElementById("hourInput").value = 3.0;
    updateHourDisplay(3.0);
    document.getElementById("daySelect").value = "Wednesday";
    document.getElementById("carsInput").value = 8;
    document.getElementById("bikesInput").value = 2;
    document.getElementById("busesInput").value = 1;
    document.getElementById("trucksInput").value = 8;
  } else if (type === "weekend") {
    document.getElementById("hourInput").value = 14.5;
    updateHourDisplay(14.5);
    document.getElementById("daySelect").value = "Saturday";
    document.getElementById("carsInput").value = 45;
    document.getElementById("bikesInput").value = 16;
    document.getElementById("busesInput").value = 5;
    document.getElementById("trucksInput").value = 3;
  }
  
  document.getElementById("carsVal").innerText = document.getElementById("carsInput").value;
  document.getElementById("bikesVal").innerText = document.getElementById("bikesInput").value;
  document.getElementById("busesVal").innerText = document.getElementById("busesInput").value;
  document.getElementById("trucksVal").innerText = document.getElementById("trucksInput").value;
  
  updateTotals();
  submitPrediction();
}

async function submitPrediction() {
  const payload = {
    hour: parseFloat(document.getElementById("hourInput").value),
    day: document.getElementById("daySelect").value,
    cars: parseInt(document.getElementById("carsInput").value),
    bikes: parseInt(document.getElementById("bikesInput").value),
    buses: parseInt(document.getElementById("busesInput").value),
    trucks: parseInt(document.getElementById("trucksInput").value)
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.status === "success") {
      const box = document.getElementById("predictionResult");
      box.classList.add("active");

      const badge = document.getElementById("resBadge");
      badge.className = `cluster-badge badge-c${data.cluster}`;
      badge.innerText = `Cluster ${data.cluster}`;

      document.getElementById("resPersona").innerText = data.persona;
      document.getElementById("resDistance").innerText = data.distance_to_center.toFixed(2);
      document.getElementById("resDescription").innerText = data.description;

      const barsContainer = document.getElementById("resSituationBars");
      barsContainer.innerHTML = "";
      
      const colors = {
        heavy: "#ef4444",
        high: "#f59e0b",
        normal: "#3b82f6",
        low: "#10b981"
      };

      for (const [sit, prob] of Object.entries(data.situation_probs)) {
        const row = document.createElement("div");
        row.style.marginBottom = "0.25rem";
        row.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.15rem;">
            <span style="text-transform: capitalize;">${sit}</span>
            <span><strong>${prob.toFixed(1)}%</strong></span>
          </div>
          <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background-color: ${colors[sit] || '#3b82f6'}; width: ${prob}%; height: 100%;"></div>
          </div>
        `;
        barsContainer.appendChild(row);
      }
    }
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  const hourInput = document.getElementById("hourInput");
  if (hourInput) {
    updateHourDisplay(hourInput.value);
    updateTotals();
  }
});

