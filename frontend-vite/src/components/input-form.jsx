export default function InputForm({prompt, setPrompt, submitPrompt, isResponseGenerating}) {
  return (
    <form className="relative flex" onSubmit={e => submitPrompt(e)}>
      <input onChange={e => setPrompt(e.target.value)}
        value={prompt} type="text" disabled={isResponseGenerating}
        className="p-4 w-full border-l border-y rounded-l-full outline-none"
        placeholder="Hey there, how're you doing today?" />
      <button type='submit'
        className="pr-4 rounded-r-full border-y border-r bg-white transition">
        ❯
      </button>
    </form>
  );
}
