const map = L.map('map').setView([31.6295, -7.9811], 15);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

let ambulanceMarkers={}, hospitalMarkers={}, accidentMarker=null, selectedAmbulance=null, selectedHospital=null, blinkInterval=null;

function updateStatus(text){document.getElementById("statusBox").innerText="Statut: "+text;}

function loadData(){
    fetch("/data").then(r=>r.json()).then(data=>{
        data.ambulances.forEach(a=>{
            if(!ambulanceMarkers[a.id]) ambulanceMarkers[a.id]=L.marker([a.lat,a.lng]).addTo(map).bindPopup("Ambulance #"+a.id);
            else ambulanceMarkers[a.id].setLatLng([a.lat,a.lng]);
        });
        const select=document.getElementById("hospitalSelect");
        if(select.options.length===0){
            data.hospitals.forEach(h=>{
                hospitalMarkers[h.id]=L.marker([h.lat,h.lng]).addTo(map).bindPopup(h.name);
                const opt=document.createElement("option");
                opt.value=h.id; opt.text=h.name; select.appendChild(opt);
            });
        }
        if(accidentMarker) map.removeLayer(accidentMarker);
        accidentMarker=null;
        if(data.accident){
            let color="green"; if(data.accident.severity==="medium") color="orange"; if(data.accident.severity==="high") color="red";
            accidentMarker=L.circle([data.accident.lat,data.accident.lng],{radius:60,color:color,fillOpacity:0.7}).addTo(map)
            .bindPopup("🚨 Accident: "+data.accident.severity);
            if(blinkInterval) clearInterval(blinkInterval);
            let visible=true;
            blinkInterval=setInterval(()=>{accidentMarker.setStyle({opacity:visible?1:0.3}); visible=!visible;},500);
            updateStatus("Accident détecté !");
        }
    });
}

function receiveAlert(){fetch("/generate_road_accident").then(r=>r.json()).then(()=>loadData());}

function optimizeAmbulance(){
    fetch("/optimize",{method:"POST"}).then(r=>r.json()).then(data=>{
        selectedAmbulance=data.ambulance.id;
        selectedHospital=data.hospital.id;
        document.getElementById("hospitalSelect").value=selectedHospital;
        updateStatus("Ambulance #"+selectedAmbulance+" sélectionnée | Hôpital optimal: "+data.hospital.name+" | ETA: "+data.eta_minutes+" min");
    });
}

function animateMarker(marker, destLat, destLng, callback=null){
    const steps=100, interval=15;
    const start=marker.getLatLng();
    let i=0;
    const move=setInterval(()=>{
        i++; const lat=start.lat+(destLat-start.lat)*(i/steps);
        const lng=start.lng+(destLng-start.lng)*(i/steps);
        marker.setLatLng([lat,lng]);
        if(i>=steps){clearInterval(move); if(callback) callback();}
    },interval);
}

function dispatchAmbulance(){
    if(!selectedAmbulance || !selectedHospital){alert("Optimise d'abord !"); return;}
    if(!accidentMarker){alert("Aucun accident détecté !"); return;}
    const ambMarker=ambulanceMarkers[selectedAmbulance];
    const accidentPos=accidentMarker.getLatLng();
    const hospitalPos=hospitalMarkers[selectedHospital].getLatLng();
    updateStatus("Ambulance en route vers accident...");
    animateMarker(ambMarker,accidentPos.lat,accidentPos.lng,()=>{
        updateStatus("Arrivée sur accident. Transport vers hôpital...");
        animateMarker(ambMarker,hospitalPos.lat,hospitalPos.lng,()=>{
            updateStatus("Patient transféré à hôpital ✅");
            fetch("/dispatch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ambulance_id:selectedAmbulance})});
            loadData();
        });
    });
}

loadData();