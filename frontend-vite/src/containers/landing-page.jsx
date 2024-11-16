import { useState } from 'react';
import PeoplePlusAILogo from "../assets/people-plus-ai-logo.svg";
import SukoonLogo from "../assets/sukoon-logo.png";
import NewChat from '../components/new-chat';
import Conversation from '../components/conversation';

export default function LandingPage() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [isResponseGenerating, setIsResponseGenerating] = useState(false);

  const submitPrompt = async e => {
    e.preventDefault();

    setIsResponseGenerating(true);

    setPrompt("");

    setMessages(messages => [...messages, {message: prompt, isResponse: false}]);

    try {
      const response = await fetch("https://sukoon-api.pplus.ai/query", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: prompt }),
      });

      const data = await response.json();

      setMessages(messages => [...messages, {message: data.output, isResponse: true}]);

      setIsResponseGenerating(false);
    } catch (error) {
      console.error('Error:', error);
    }
  }

  return (
    <div className="p-5">
      <a href='/'><img src={SukoonLogo} className="w-16 md:w-24" /></a>
      <a href='https://peopleplus.ai/' target='_blank'><img src={PeoplePlusAILogo} className="absolute float-right top-0 right-0 w-48" /></a>
      {
        messages.length > 0 ?
          <Conversation isResponseGenerating={isResponseGenerating} prompt={prompt}
            submitPrompt={submitPrompt} setPrompt={setPrompt} messages={messages} />
        :
          <NewChat prompt={prompt} setPrompt={setPrompt} submitPrompt={submitPrompt} />
      }
    </div>
  );
}
