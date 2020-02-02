const express = require('express');
const fs = require('fs');
const multer = require('multer');
const shell = require('shelljs')


var storage = multer.diskStorage({
	destination: function (req, file, cb) {
	  cb(null, __dirname + '/uploads')
	},
	filename: function (req, file, cb) {
	  cb(null, "recent.jpg")
	}
  })



const upload = multer({storage : storage});

const app = express();
const PORT = 3000;

app.use(express.static('public'));
app.use(express.static('uploads'));
app.post('/upload', upload.single('photo'), (req, res) => {
	if(req.file) {
		shell.exec('python ./uploads/new.py');
		
		setTimeout(()=>{
		fs.access(__dirname + '/uploads/output.jpg', fs.F_OK, (err) => {
			if (err) {
			  console.error(err)
			  return
			}
			fs.unlink(__dirname + '/uploads/output.jpg');
		});
		fs.access(__dirname + '/uploads/recent.jpg', fs.F_OK, (err) => {
			if (err) {
			  console.error(err)
			  return
			}
			fs.unlink(__dirname + '/uploads/recent.jpg');
		});
		},3000);

		res.writeHead(302, {'Location': 'http://localhost:3000/index2.html'});
		res.end();
    }
    else throw 'error';
});

app.listen(PORT, () => {
    console.log('Listening at ' + PORT );
});
