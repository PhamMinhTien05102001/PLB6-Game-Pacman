import { useStore } from '../../../components/StoreContext';
import { MAX_POWER } from '../../../model/detectCollisions';

export const useGameState = () => {
  const store = useStore();
  const { game } = store;
  if (game.atePills === MAX_POWER) alert('Congrats you win the game');
};
