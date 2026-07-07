let selectedModel = "segment";

function selectModel(model) {

    selectedModel = model;

    document.querySelectorAll(".model-buttons button").forEach(btn => {
        btn.classList.remove("selected");
    });

    document.getElementById(model + "Btn").classList.add("selected");
}

function runModel() {

    const file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const outputImg = document.getElementById("outputImg");
    const maskImg = document.getElementById("maskImg");
    const maskCard = document.getElementById("maskCard");

    maskCard.style.display = "none";
    maskImg.src = "";

    const formData = new FormData();
    formData.append("image", file);
    formData.append("model", selectedModel);

    outputImg.src = "";

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        if (data.error) {
            alert(data.error);
            return;
        }

        outputImg.src = "data:image/jpeg;base64," + data.result;

        if (data.mask) {

            maskImg.src = "data:image/jpeg;base64," + data.mask;
            maskCard.style.display = "block";

        } else {

            maskCard.style.display = "none";

        }

    })
    .catch(err => {

        console.error(err);
        alert("An error occurred while running the segmentation.");

    });

}

selectModel("segment");
