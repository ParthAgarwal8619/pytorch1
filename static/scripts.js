function upload() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Prediction: " + data.prediction;
    });
}