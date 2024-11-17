import React, { useState, useRef, useEffect } from 'react';
import './App.css';
// import Menu from './components/Menu';
import Feedback from './components/Feedback';
import Recommand from './components/Recommand';

function App() {
  const [input, setInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const thumbUpBtn = useRef();
  const thumbDownBtn = useRef();
  const messagesEndRef = useRef(null);

  // Modal
  // State to control modal visibility
  const [showModal, setShowModal] = useState(false);
  const [status, setStatus] = useState({ status: "wait", message: "" });

  // Functions to open and close the modal
  const handleOpenModal = (action) => {
    if (status.status !== "success") {
      if (action === 'Like') {
        thumbUpBtn.current.classList.add('text-warning')
        thumbDownBtn.current.classList.remove('text-warning')
      } else if (action === 'Dislike') {
        thumbDownBtn.current.classList.add('text-warning')
        thumbUpBtn.current.classList.remove('text-warning')
      }
    } else {
      setFeedback(action)
    }
    setShowModal(true)
  };


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async (message) => {
    let inputMsg = message ? message : input;
    if (inputMsg.trim() === '') return;

    setMessages([...messages, { text: inputMsg, user: true }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch("https://sukoon-api.pplus.ai/query", { // http://0.0.0.0:8001/query
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: inputMsg }),
      });

      const data = await response.json();
      // console.log(data)
      setMessages(prevMessages => [...prevMessages, { text: data.output, user: false }]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">

      <div className="chat-container" style={{ display: "flex" }}>
        {/* <Menu /> */}
        {/* If Menu Show in page the .chatConv width is 70% other wise width is 100% */}
        <div class={`chatConv justify-content-center pt-4 pe-4 ${!messages.length ? "align-items-center d-flex" : ""}`} style={{ height: "70vh", overflow: "scroll", overflowX: "auto" }}>
          <div class="col-6 mx-auto">
            <div>
              {messages.map((message, index) => (
                <div className={`m-4 ${message.user ? 'text-end' : 'text-start'}`} key={index}><span className={`p-2 me-3 my-1 rounded-3 ${message.user ? 'bg-primary text-light' : ''}`} style={{ display: 'inline-block' }}>{message.text}</span></div>
              ))}
              {isLoading && (
                <div className="loading-container">
                  <div className="loading-bar"></div>
                </div>
              )}
            </div>
            <div class="chatAction d-flex justify-content-around bd-highlight">
              {!messages.length ? <Recommand recommandSend={handleSend}/> : ""}
              {/* <Recommand /> */}
              <div class="p-2 bd-highlight query-action" style={{ position: "fixed", bottom: "40px" }}>
                <div class="input-group mb-3" style={{ width: "400px" }}>
                  <div class="form-floating"><input type="text" class="w-100 h-100" id="floatingInputGroup1"
                    placeholder="Type your message..." value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} /></div><span
                      class="input-group-text d-flex justify-content-center submit" onClick={handleSend}><i
                        class="fa-solid fa-arrow-up"></i>
                  </span>
                </div>
                <p style={{ textAlign: "center", fontSize: "small" }}>By using this, you give your consent that you'e read the <a href="https://peopleplusai.github.io/Sukoon/disclaimer.html" target='_blank' rel="noopener noreferrer">disclaimer.</a></p>
                <p style={{ textAlign: "center", fontSize: "small" }}>How did you like the conversation? Please provide feedback. &nbsp;&nbsp;&nbsp;&nbsp;<i className="like fa-solid fa-thumbs-up fs-5" onClick={() => handleOpenModal("Like")} ref={thumbUpBtn}></i>&nbsp;&nbsp;&nbsp;<i className="dislike fa-solid fa-thumbs-down fs-5" onClick={() => handleOpenModal("Dislike")} ref={thumbDownBtn}></i></p>
              </div>
            </div>
          </div>
          <p ref={messagesEndRef} style={{ display: "ruby-text" }} />
        </div>
      </div>
      <Feedback show={showModal} setShowModal={setShowModal} act={feedback} thumbDownBtn={thumbDownBtn} thumbUpBtn={thumbUpBtn} setStatus={setStatus} status={status} />
    </div>
  );
}

export default App;
