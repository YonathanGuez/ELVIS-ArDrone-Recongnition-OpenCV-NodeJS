
//Utilisation de la bibliothèque ar drone
var arDrone = require('ar-drone');

//Utilisation de la bibliothèque socket.io qui s'occupe de tout ce qui est temps réel
var io = require('socket.io-client'),

//On se connecte au server local 3111
socket = io.connect('localhost', { port: 3111});
//On affiche dans la console la connection 
socket.on('connect', function () { console.log("socket connected"); });

//Socket emet une image de nom : drone
socket.emit('image', { image: 'drone' });

//On créé un client de notre drone
var client = arDrone.createClient();

//On demande au client d'enregistrer des images de type PNG
client.getPngStream()
	//En cas d'erreur, l'afficher dans la console
    .on('error', console.log)
	//En cas de donnée, socket emettra sur le server les images de notre camera
    .on('data', function(frame) {
        socket.emit('image', { image: frame });
    });
	