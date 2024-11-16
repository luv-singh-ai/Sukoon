export default function InputForm({prompt, setPrompt, submitPrompt, isResponseGenerating}) {
  return (
    <>
      <form className="relative flex" onSubmit={e => submitPrompt(e)}>
        <input onChange={e => setPrompt(e.target.value)}
          value={prompt} type="text" disabled={isResponseGenerating}
          className="p-4 w-full border-l-2 border-y-2 rounded-l-full outline-none"
          placeholder="Hey there, how're you doing today?" />
        <button type='submit'
          className="pr-8 rounded-r-full border-y-2 border-r-2 bg-white transition">
          ❯
        </button>
      </form>
      <div className="mt-1 md:mt-3">
        <p className="text-center text-xs md:text-sm text-slate-400">
          By using this, you give your consent that you have read the <a className="text-blue-500" href="https://peopleplusai.github.io/Sukoon/disclaimer.html" target='_blank' rel="noopener noreferrer">disclaimer</a>.
        </p>
      </div>
    </>
  );
}
