import { useCallback, useEffect } from 'react';
import { useStore } from '../../../components/StoreContext';
import { useGame } from '../../../components/StoreContext';

/* eslint-disable  react-hooks/exhaustive-deps */
export const useKeyboardActions = (): void => {
  const store = useStore();

  const onKeyDown = useCallback((event: KeyboardEvent) => {
    const { game } = store;
    const pressedKey = event.key;
    const pacMan = game.pacMan;

    switch (pressedKey) {
      case 'ArrowLeft':
        pacMan.nextDirection = 'LEFT';
        break;
      case 'ArrowRight':
        pacMan.nextDirection = 'RIGHT';
        break;
      case 'ArrowUp':
        pacMan.nextDirection = 'UP';
        break;
      case 'ArrowDown':
        pacMan.nextDirection = 'DOWN';
        break;
      case 'x':
        if(game.atePills === 150){
          game.killedGhosts = 0;
          game.pacMan.send('ENERGIZER_EATEN');
          for (const ghost of game.ghosts) {
            ghost.send('ENERGIZER_EATEN');
          }
          game.atePills = 0;
        }
        break;
      case ' ':
        game.gamePaused = !game.gamePaused;
        break;
      default:
        break;
    }
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, []);
};
