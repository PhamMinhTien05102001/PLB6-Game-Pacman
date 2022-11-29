import { useCallback, useEffect } from 'react';
import { useStore } from '../../../components/StoreContext';

import { MAX_POWER } from '../../../model/detectCollisions';

/* eslint-disable  react-hooks/exhaustive-deps */
export const useKeyboardActions = (playByCamera: boolean): void => {
  const store = useStore();

  const onKeyDown = useCallback((event: KeyboardEvent) => {
    const { game } = store;
    const pressedKey = event.key;
    const pacMan = game.pacMan;

    switch (pressedKey) {
      case 'a':
        pacMan.nextDirection = 'LEFT';
        break;
      case 'd':
        pacMan.nextDirection = 'RIGHT';
        break;
      case 'w':
        pacMan.nextDirection = 'UP';
        break;
      case 's':
        pacMan.nextDirection = 'DOWN';
        break;
      case 'x':
        game.pacMan.send('ENERGIZER_EATEN');
        for (const ghost of game.ghosts) {
          ghost.send('ENERGIZER_EATEN');
        }
        break;
      case ' ':
        if (game.atePills === MAX_POWER) {
          game.killedGhosts = 0;
          game.pacMan.send('ENERGIZER_EATEN');
          for (const ghost of game.ghosts) {
            ghost.send('ENERGIZER_EATEN');
          }
          game.atePills = 0;
        }
        break;

      default:
        break;
    }
  }, []);

  const onPause = useCallback((event: KeyboardEvent) => {
    const { game } = store;
    const pressedKey = event.key;

    switch (pressedKey) {
      case 'p':
        game.gamePaused = !game.gamePaused;
        break;
      default:
        break;
    }
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', onPause);
    if (!playByCamera) {
      window.addEventListener('keydown', onKeyDown);
    } else {
      window.removeEventListener('keydown', onKeyDown);
    }

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [playByCamera]);
};
