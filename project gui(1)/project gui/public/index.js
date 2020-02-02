
$("#figr").click( printPosition)
$("#figr2").click(printPosition)

function getPosition(e) {
  var rect = e.target.getBoundingClientRect();
  var x = e.clientX - rect.left;
  var y = e.clientY - rect.top;
  return {
    x,
    y
  }
}

function printPosition(e) {
  let position = getPosition(e);
  let rect = "rect(0px,"+ position.x +"px, 1000px, 0px)";
  document.getElementById("figr").style.clip =  rect;
}


function check(){
	document.getElementById("img2").style.display = "block";
}
