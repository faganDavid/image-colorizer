function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#grayScaleImg").attr("src", e.target.result).width(300).height(300);
    };

    reader.readAsDataURL(input.files[0]);
  }
}

function show_grey(obj) {
  $(obj).addClass("grey-scale");
}
function show_original(obj) {
  $(obj).removeClass("grey-scale");
}

// define URL and for element
const url = "/predictimage";
const form = document.querySelector("form");

// add event listener
form.addEventListener("submit", (e) => {
  // disable default action
  e.preventDefault();

  // collect files
  const files = document.querySelector("[name=file]").files;
  const formData = new FormData();
  formData.append("file", files[0]);

  // post form data
  const xhr = new XMLHttpRequest();

  // log response
  // Setup our listener to process compeleted requests
  xhr.onreadystatechange = function () {
    // Only run if the request is complete
    if (xhr.readyState !== 4) return;

    // Process our return data
    if (xhr.status >= 200 && xhr.status < 300) {
      // What do when the request is successful
      //console.log("I got the image");

      let data = JSON.parse(xhr.responseText);

      //console.log(data.image_url);

      var img = new Image();
      img.src = "data:image/jpeg;base64," + data.image_url;

      img.onload = function () {
        const canvasMask = document.getElementById("myCanvas");
        const ctxcanvasMask = canvasMask.getContext("2d");
        ctxcanvasMask.drawImage(img, 0, 0);
      };
    }
  };

  // create and send the reqeust
  xhr.open("POST", url);
  xhr.send(formData);
});
