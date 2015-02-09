

 //Utilisation de la bibliothèque express qui gère
 //tout ce qui est framework (= bibliothèque)
var express = require('express')
var app = express()

//Creation d'un server de type http pour app
var server = require("http").createServer(app)

//Indique que le dossier public contient des fichiers statiques : images 
app.use(express.static(__dirname + '/public'));

//Require 2 fichiers : camera-feed et controller
require("./drone/camera-feed");
require("./drone/controller");

//Ecoute sur le port 3000
app.listen(3000);


