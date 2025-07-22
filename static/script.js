const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

    // Setup canvas
function setupCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
    }

setupCanvas();

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
        drawing = true;
        const pos = getMousePos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    });

canvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const pos = getMousePos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    });

canvas.addEventListener("mouseup", () => {
        drawing = false;
        ctx.closePath();
    });

canvas.addEventListener("mouseleave", () => {
        drawing = false;
        ctx.closePath();
    });


function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
}

function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setupCanvas();
        document.getElementById("result").innerText = "Draw a digit";
        document.getElementById("probs").innerHTML = "";
}

function predict() {
        const image = canvas.toDataURL("image/png");
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: image })
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerText = `Prediction: ${data.digit}`;
            showProbs(data.probs);
        });
}

function showProbs(probs) {
        const container = document.getElementById("probs");
        container.innerHTML = "";
        probs.forEach((p, i) => {
            const label = document.createElement("div");
            label.className = "bar-label";
            label.innerHTML = `<span>${i}</span><span>${(p * 100).toFixed(2)}%</span>`;

            const bar = document.createElement("div");
            bar.className = "bar";
            bar.style.width = `${(p * 100).toFixed(2)}%`;

            container.appendChild(label);
            container.appendChild(bar);
        });
}