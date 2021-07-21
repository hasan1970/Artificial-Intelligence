import random
import time

class Blackjack():


    def __init__(self):


        self.player_cards=[]
        self.opponent_cards=[]
    
        player_card_1=random.randint(1,10)
        opponent_card_1=random.randint(1,10)
        self.player_cards.append(player_card_1)
        self.opponent_cards.append(opponent_card_1)
    
        player_card_2=random.randint(1,10)
        opponent_card_2=random.randint(1,10)
        self.player_cards.append(player_card_2)
        self.opponent_cards.append(opponent_card_2)

        self.player_turn=0
        self.winner=None
        self.available_actions=['h','s']

   
    @classmethod
    def other_player(cls, player):

            return 0 if player==1 else 1
        
    def switch_player(self):
            self.player_turn=Blackjack.other_player(self.player_turn)

    def check_for_winner(a,b):
           # print("In Blackjack, in check4winner")

            #1 for winning, -1 for losing, 0 for Tie
            if sum(a)==21 and sum(b)==21:
                return 0

            if sum(a)==21:
                return 1
            elif sum(b)==21:
                return -1
        
            elif sum(a)>21:
                return -1
                
            elif sum(b)>21:
                return 1


            elif sum(a)<21 and sum(b)<21:

                if sum(a)>sum(b):
                    return 1
                    
                elif sum(a)==sum(b):
                    return 0
                else:
                    return -1
    
                    
   

    def playing_a_move(self, action):
           # print("In Blackjack, in playingaMove")
            move=action

            if self.winner is not None:
                raise Exception("Game is already over!")
            #elif sum_of_cards<0:
             #   raise Exception("Invalid card")
            elif move not in ['h','s']:
                raise Exception('Invalid move!')


            #Updating the cards
            if move=='h':
                card=random.randint(1,10)
                #If its the humans turn
                if self.player_turn==0:
                    self.player_cards.append(card)
                    #If the cards are greater than 21, then the other player wins. Human player is assigned 0 by default
                    if sum(self.player_cards)>21:
                        self.winner=1
                    #Human winner
                    if sum(self.player_cards)==21:
                        self.winner=0

                #if its the AI's turn
                elif self.player_turn == 1:
                    self.opponent_cards.append(card)
                    #If the cards are greater than 21, then the other player wins. Human player is assigned 0 by default
                    if sum(self.opponent_cards)>21:
                        self.winner=0
                    #AI turn
                    if sum(self.opponent_cards)==21:
                        self.winner=1
            
            elif move=='s':

                result=Blackjack.check_for_winner(self.player_cards,self.opponent_cards)
                if result==1:
                    self.winner = 0 #Human Won
                elif result == -1:
                    self.winner = 1 #AI won
                elif result == 0:
                    self.winner = 'T'
                else:
                    raise Exception("Result returned wrong")

            self.switch_player()


class BlackjackAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is the sum of the current card the person holds eg. [3,4,5] would have the state as 12
         - `action` is either 'h' (for hit) or 's' (for stand)

        """
        #print("In BlackjackAI, in init")
        self.q = dict()
        self.q[((21,),'s')]=1
        self.q[((20,),'s')]=1
        self.q[((19,),'s')]=1
        self.q[((18,),'s')]=1
        self.alpha=alpha
        self.epsilon=epsilon
        self.available_actions=['h','s']


    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        #print("In BlackjackAI, in update")
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

            

    def get_q_value(self,state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        
        """
        #print("In BlackjackAI, in get q value")
        #print(state)
        
        Tstate=tuple([state])
        
        if (Tstate, action) in self.q:
            return self.q[(Tstate,action)]
        else:
            return 0


    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        
        """
        
        Tstate=tuple([state])
        alpha = self.alpha
        new = reward + future_rewards

        result = old_q + ( alpha * (new-old_q) )
        self.q[(Tstate,action)]=result
        #print(self.q)
        #print('=============')
        #print(state,Tstate,result,action)


    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.
        """
        #print("In BlackjackAI, in best future rewrd")
        values=[]
        Tstate=tuple([state])
        for action in self.available_actions:
            combo=(Tstate,action)
            if combo in list(self.q.keys()):
                value=self.q[combo]
                values.append(value)
            else:
                values.append(0)
    
        result=max(values)
        return result

    def best_action(self,state):
        #print("In BlackjackAI, in best action")
        actions={}
        #print(state)
        Tstate=tuple([state])
        
        
        for action in self.available_actions:
            combo=(Tstate,action)   
            if combo in list(self.q.keys()):
                value=self.q[(Tstate,action)]
                actions[action]=value

        if actions == {}:      
            return action

        sort = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        topaction=sort[0][0]
        return topaction

    def choose_action(self, state, epsilon=True):

        #print("In BlackjackAI, in choose action")
        
        if epsilon is False:
            bestaction=self.best_action(state)
            return bestaction
            

        else:
          
           actionslist=[]
           eps=self.epsilon
           bestaction=self.best_action(state)
        
           for action in self.available_actions:
               actionslist.append(action)
           randaction=random.choice(actionslist)

           list_of_choices=[bestaction,randaction]
           choice=random.choices(list_of_choices,weights=[eps,1-eps],k=1)

           return choice[0]
            
        
def train(n):
    """
    Train an AI by playing `n` games against itself.
    """
    #print(",, in train")
    player = BlackjackAI()

    # Play n games
    for i in range(n):
        #print(f"Playing training game {i + 1}")
        game = Blackjack()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            if game.player_turn==0:
                state = sum(game.player_cards)
                action = player.choose_action(game.player_cards)
            elif game.player_turn == 1:
                state = sum(game.opponent_cards)
                action = player.choose_action(game.opponent_cards)

            # Keep track of last state and action
            last[game.player_turn]["state"] = state
            last[game.player_turn]["action"] = action

            # Make move
            game.playing_a_move(action)
            if game.player_turn==0:
                new_state=sum(game.player_cards)
            elif game.player_turn==1:
                new_state=sum(game.opponent_cards)

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player_turn]["state"],
                    last[game.player_turn]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player_turn]["state"] is not None:
                player.update(
                    last[game.player_turn]["state"],
                    last[game.player_turn]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return player

def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    #print(",,In play")

    # If no player order set, choose human's order randomly
    human_player=0


    # Create new game
    game = Blackjack()

    # Game loop
    while True:

        # Print cards
        print()
        print("Your cards: {}, total: {} ".format(game.player_cards,sum(game.player_cards)))
        print()

        # Compute available actions
        available_actions = game.available_actions
        time.sleep(1)

        # Let human make a move
        if game.player_turn == human_player:
            print("Your Turn")
            while True:
                turn = input("Do you want to hit (h) or stand (s)? ")
                if turn in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            turn = ai.choose_action(sum(game.opponent_cards), epsilon=False)
            print()
            if turn=='h':
                print('AI chooses to hit!')
            elif turn=='s':
                print('AI chooses to stand!')
        

        # Make move
        game.playing_a_move(turn)

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")

            if game.winner==0:
                winner="Human"
            elif game.winner==1:
                winner="AI"
            elif game.winner=='T':
                winner="No One. Its a tie."
   
            print('Your cards were {}, total: {}'.format(game.player_cards,sum(game.player_cards)))
            print('The AI\'s cards were {}, total: {}'.format(game.opponent_cards,sum(game.opponent_cards)))
            print(f"Winner is {winner}")
            return

