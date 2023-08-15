const WebSocket = require('ws');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});


// All the events that can be sent to the server.
class Interrupt {
  constructor(key = null) {
    this.event = 'interrupt';
    this.key = key;
  }
}

class Close {
  constructor(key = null) {
    this.event = 'close';
    this.key = key;
  }
}

class ApiChatContent {
  constructor(conversation = null, settings = null, key = null, id = null) {
    this.event = 'user_message';
    this.conversation = conversation;
    this.settings = settings;
    this.key = key;
    this.id = id;
  }
}

// Settings for the API.
class ApiSettings {
  constructor(language = null) {
    this.language = language;
  }
}

rl.question('Paste your health question: ', (question) => {
  rl.question('Paste your API key (email founders@medisearch.io if do not have one): ', (api_key) => {
    const ws = new WebSocket('wss://public.backend.medisearch.io:443/ws/medichat/api');
    ws.on('open', function open() {
      const settings = new ApiSettings('English');
      const chatContent = new ApiChatContent([question],
        settings,
        api_key,
        generateID());
      ws.send(JSON.stringify(chatContent));
    });


    // Prepare for receiving the request
    ws.on('message', function incoming(data) {
      const strData = data.toString('utf8');
      const jsonData = JSON.parse(strData);

      if (jsonData.event === "articles") {
        console.log("Got articles");
      } else if (jsonData.event === "llm_response") {
        console.log("Got llm response");
      } else if (jsonData.event === "error") {
        console.log("Got error");
      }
      console.log(jsonData);
    });

    ws.on('error', function error(err) {
      console.log('WebSocket Error:', err);
    });

  });
});

function generateID() {
  var id = '';
  var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

  for (var i = 0; i < 32; i++) {
    id += characters.charAt(Math.floor(Math.random() * characters.length));
  }

  return id;
}