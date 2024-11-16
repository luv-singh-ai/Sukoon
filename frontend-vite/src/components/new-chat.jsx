import React from 'react'
import InputForm from './input-form';

export default function NewChat({prompt, setPrompt, submitPrompt}) {
  return (
    <>
      <div style={{height: "80vh"}} className="h-screen flex items-center justify-center">
        <div className='w-full md:w-10/12 lg:w-9/12 xl:w-1/2'>
          <InputForm prompt={prompt} setPrompt={setPrompt} submitPrompt={submitPrompt} />
        </div>
      </div>
    </>
  );
}
