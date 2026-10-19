const socket = io();

let map = L.map('map').setView([31.63, -8.00], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png')
.addTo(map);

let ambulanceMarker = L.marker([31.63, -8.00]).addTo(map);

map.on('click', function(e){

    fetch('/generate_accident', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            lat: e.latlng.lat,
            lng: e.latlng.lng,
            severity: 4
        })
    });

});

function sendAmbulance(){
    fetch('/dispatch_ambulance', {method:'POST'})
}

socket.on("eta_update", function(data){
    document.getElementById("eta").innerHTML =
        "ETA: " + data.eta + " secondes";
});

socket.on("ambulance_position", function(data){
    ambulanceMarker.setLatLng([data.x/100000, data.y/100000]);
});

socket.on("accident_created", function(data){
    document.getElementById("severity").innerHTML =
        "Gravité: " + data.severity;
});