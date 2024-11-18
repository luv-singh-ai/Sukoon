export default function InputForm({prompt, setPrompt, submitPrompt, isResponseGenerating, showSuggestions}) {
  return (
    <>
      {
        showSuggestions === true && <div className="flex flex-wrap justify-center gap-3 md:gap-4">
          <div className="whitespace-nowrap p-2 md:p-3 px-4 md:px-6 text-sm md:text-base rounded-full border-5 outline-5 shadow-md cursor-pointer bg-[#fff9f5] hover:bg-[#ffeee3] transition duration-300"
            onClick={e => submitPrompt(e.target.innerHTML)}>
            I need to vent
          </div>
          <div className="whitespace-nowrap p-2 md:p-3 px-4 md:px-6 text-sm md:text-base rounded-full border-5 outline-5 shadow-md cursor-pointer bg-[#f5f5ff] hover:bg-[#e3e3ff] transition duration-300"
            onClick={e => submitPrompt(e.target.innerHTML)}>
            I'm feeling overwhelmed
          </div>
          <div className="whitespace-nowrap p-2 md:p-3 px-4 md:px-6 text-sm md:text-base rounded-full border-5 outline-5 shadow-md cursor-pointer bg-[#f5f5ff] hover:bg-[#e3e3ff] transition duration-300"
            onClick={e => submitPrompt(e.target.innerHTML)}>
            I'm feeling extremely stressed
          </div>
          <div className="whitespace-nowrap p-2 md:p-3 px-4 md:px-6 text-sm md:text-base rounded-full border-5 outline-5 shadow-md cursor-pointer bg-[#fff9f5] hover:bg-[#ffeee3] transition duration-300"
            onClick={e => submitPrompt(e.target.innerHTML)}>
            Help me understand my feelings
          </div>
          <div className="whitespace-nowrap p-2 md:p-3 px-4 md:px-6 text-sm md:text-base rounded-full border-5 outline-5 shadow-md cursor-pointer bg-[#f5fcf2] hover:bg-[#e8ffde] transition duration-300"
            onClick={e => submitPrompt(e.target.innerHTML)}>
            I don't know how to deal with this situation
          </div>
        </div>
      }
      <form className="relative flex mt-3 md:mt-4" onSubmit={e => {e.preventDefault(); submitPrompt(prompt);}}>
        <input onChange={e => setPrompt(e.target.value)}
          value={prompt} type="text" disabled={isResponseGenerating}
          className="p-3 px-4 md:p-4 w-full border-l-2 border-y-2 rounded-l-full outline-none"
          placeholder="Hey there, how're you doing today?" />
        <button type='submit'
          className="pr-8 rounded-r-full border-y-2 border-r-2 bg-white transition">
          ❯
        </button>
      </form>
      <div className="mt-2 md:mt-3">
        <p className="text-center text-xs md:text-sm text-slate-400">
          By using this, you give your consent that you have read the <a className="text-blue-500" href="https://peopleplusai.github.io/Sukoon/disclaimer.html" target='_blank' rel="noopener noreferrer">disclaimer</a>.
        </p>
      </div>
    </>
  );
}
