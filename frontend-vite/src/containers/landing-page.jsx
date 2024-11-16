import { useState } from 'react';
import PeoplePlusAILogo from "../assets/people-plus-ai-logo.svg";
import SukoonLogo from "../assets/sukoon-logo.png";
import NewChat from '../components/new-chat';
import Conversation from '../components/conversation';
import { BACKEND_ENDPOINT } from "../utils/envs";
import { RemoveTrailingSlash } from '../utils/url';

export default function LandingPage() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [isResponseGenerating, setIsResponseGenerating] = useState(false);

  const submitPrompt = async e => {
    e.preventDefault();

    setIsResponseGenerating(true);

    setPrompt("");

    setMessages(messages => [...messages, {message: prompt, isResponse: false}]);

    const backendUrl = `${RemoveTrailingSlash(BACKEND_ENDPOINT)}/query`

    try {
      const response = await fetch(backendUrl, {
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
      <div className='h-[10vh]'>
        <a className='inline-block' href='/'>
          <img src={SukoonLogo} className="w-16 md:w-24" />
        </a>
        <a href='https://peopleplus.ai/' target='_blank'>
          <img src={PeoplePlusAILogo} className="absolute float-right top-0 right-0 w-48" />
        </a>
      </div>
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
