const canvasMask = document.getElementById('myCanvas');
const ctxcanvasMask = canvasMask.getContext('2d');

function readURL(input) {
if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
        $('#blah')
            .attr('src', e.target.result)
            .width(150)
            .height(200);
    };

    reader.readAsDataURL(input.files[0]);
}
}

document.getElementById("getMask").addEventListener('click', () => {
    var xhr = new XMLHttpRequest();
    
    // Setup our listener to process compeleted requests
    xhr.onreadystatechange = function () {
        // Only run if the request is complete
        if (xhr.readyState !== 4) return;

        // Process our return data
        if (xhr.status >= 200 && xhr.status < 300) {
            // What do when the request is successful
            console.log("I got the image");
            
            let data = JSON.parse(xhr.responseText);

            console.log(data.image_url);

            var img = new Image();
            img.src = 'data:image/jpeg;base64,' + data.image_url;
            
            img.onload = function () {
                ctxcanvasMask.drawImage(img, 0, 0);
            }
        }
    };
    // Create and send a GET request
    // The first argument is the post type (GET, POST, PUT, DELETE, etc.)
    // The second argument is the endpoint URL
    var formData = new FormData();
    formData.append("file", $('#imageInput')[0]);

    xhr.open('POST', 'http://127.0.0.1:5000/predictimage');
    xhr.setRequestHeader('Content-Type', 'multipart/form-data');
    xhr.send(formData);
});

