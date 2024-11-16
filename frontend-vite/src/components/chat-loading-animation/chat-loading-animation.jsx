import Lottie from 'react-lottie';
import AnimationData from './chat-loading-animation.json';

export default function ChatLoadingAnimation() {
  const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: AnimationData,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice"
    }
  };

  return (
    <Lottie options={defaultOptions} height={40} width={40} />
  );
}
