import { observer } from 'mobx-react-lite';
import { FC } from 'react';
import './GameOver.css';
import { useGame } from '../../../components/StoreContext';
import { Message } from './Message';
import styled from 'styled-components';

export const GameWin: FC<{ className?: string }> = observer(({ className }) => {
  const game = useGame();

  const gameWinMessageVisible = game.pillsCount >= 334;
  if (game.pillsCount >= 334) {
    game.gamePaused = true;
  }
  return gameWinMessageVisible ? (
    <>
      <Message text="You Win" />
      <Button
        onClick={() => {
          window.location.reload();
        }}
      >
        Reset Game
      </Button>
    </>
  ) : null;
});
const Button = styled.button`
  font-family: Joystix;
  font-size: 14px;
  background: yellow;
  color: white;
  cursor: pointer;
  position: absolute;
  left: 170px;
  top: 370px;
  width: 220px;
  text-align: center;
`;
