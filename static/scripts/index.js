let text = document.querySelector('#translated-text')
let button = document.querySelector('#speech-btn')
let utterance = ''

if(text.value != null){
    utterance = new SpeechSynthesisUtterance(text.value);
}

button.addEventListener("click",function(){
    speechSynthesis.speak(utterance);
})

