
 //Require la bibliothèque socket.io qui s'occupe du temps reel
 //entre mon application et mon server
var io = require('socket.io').listen(3002);


io.set('log level', 1);

//Quand un client se connecte au server on fait :
io.sockets.on('connection', function (socket) {
    //Require la bibliothèque ar-drone
	var arDrone = require('ar-drone');
	//On créée notre server
    var client = arDrone.createClient();

	//S'occupe toutes les 1000 ms de la fonction
    setInterval(function(){
		//Variable battery du client (drone)
        var batteryLevel = client.battery();
		//On écrit quand un évenement se passe :
		//On écrit au serveur le nom "battery" et on récupère la valeur
		// de la battery
        socket.emit('event', { name: 'battery',value: batteryLevel});
    },1000);


	//Lors d'un evenement (qu'on on clic sur le bouton = data)
    socket.on('event', function (data) {
        //Si le nom de la data est "décoller"
		if(data.name=="takeoff"){
			//On affiche
            console.log("Browser asked Ar Drone to Take Off");
            //Decollage
			client.takeoff();
        }
        if(data.name=="spin"){
            console.log("Browser asked Ar Drone to Start Spinning");
           client.clockwise(1);
        }
        if(data.name=="stop"){
            console.log("Browser asked Ar Drone to Stay and Hover");
            client.stop();
        }
        if(data.name=="land"){
            console.log("Browser asked Ar Drone to Land");
            client.land();
        }

    });
});
