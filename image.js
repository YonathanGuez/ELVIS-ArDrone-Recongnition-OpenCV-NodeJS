

var arDrone = require('ar-drone');
var http    = require('http');
var fs 		= require('fs');


console.log('Connection !');

var pngStream = arDrone.createClient().getPngStream();


//PNG
var tick = 0;
var sto = 0;
	var lastPng;
	pngStream
	  .on('error', console.log)
	  .on('data', function(pngBuffer) {
		if (!(tick++ % 5)) {
			lastPng = pngBuffer;
			fs.writeFileSync("./img" + sto++%6 + ".png", lastPng);
			fs.writeFileSync("./currentimg.txt", sto%6);
			fs.exists("./img/work.png", function (e) 
			{
				if (!e) {
					fs.writeFileSync("./work.png", lastPng);
				}
			});
		}
	  });

//Server
var server = http.createServer(function(req, res) {
		if (!lastPng) {
			res.writeHead(200, { 'Content-Type' :'text/plain'});
			res.end('Did not receive any png data yet.');
			return;
		}
		
		res.writeHead(200, {'Content-Type': 'image/png'});
		res.end(lastPng);
	});

//Ecoute	
server.listen(8080);


