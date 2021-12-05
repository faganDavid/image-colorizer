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


const canvasMask = document.getElementById('myCanvas');
const ctxcanvasMask = canvasMask.getContext('2d');

document.getElementById("getMask").addEventListener('click', () => {

    fetch('http://127.0.0.1:5000/predictimage')
    //.then(res => res.json())
    .then(data => {
        var img = new Image();
        img.src = 'data:image/jpeg;base64,' + data.image_url;
        img.onload = () => ctxcanvasMask.drawImage(img, 0, 0);
    })
    .catch(err => alert("PROBLEM\n\n" + err));
    
});
