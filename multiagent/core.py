import numpy as np

# physical/external base state of all entites 所有实体的状态类的基类
class EntityState(object):
    def __init__(self):
        # physical position 物理位置
        self.p_pos = None
        # physical velocity 物理速度
        self.p_vel = None

# state of agents (including communication and internal/mental state)智能体的状态，包括通信状态与内外状态
class AgentState(EntityState):# 智能体状态类，该类继承了实体状态类
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance 通信方式
        self.c = None

# action of the agent 智能体的动作类
class Action(object):
    def __init__(self):
        # physical action 物理动作，物理动作的性质是一个智能体对自己施加的力
        self.u = None
        # communication action 通信动作
        self.c = None

# properties and state of physical world entity 实体的物理属性与状态，这个类既作为其他实体的基类，又作为地标类
class Entity(object):
    def __init__(self):
        # name 实体名称
        self.name = ''
        # properties: 实体的各属性：实体的大小
        self.size = 0.050
        # entity can move / be pushed 实体是否能被推开
        self.movable = False
        # entity collides with others 实体是否会与其他实体产生碰撞
        self.collide = True
        # material density (affects mass) 实体的密度，密度将会影响实体的质量
        self.density = 25.0
        # color 实体的颜色
        self.color = None
        # max speed and accel 实体的最大速度与加速度
        self.max_speed = None
        self.accel = None
        # state 实体的状态，这个属性是一个对象
        self.state = EntityState()
        # mass 实体的初始质量
        self.initial_mass = 1.0
    # 一个查询函数，返回实体的质量
    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities 作为基类的实体类的定义同时也是地标类的定义
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities 智能体的属性
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default 智能体默认是可以被推动的
        self.movable = True
        # cannot send communication signals 智能体可以进行通信
        self.silent = False
        # cannot observe the world 智能体可以对环境进行观察
        self.blind = False
        # physical motor noise amount 智能体是否存在移动噪声，移动噪声是智能体移动时的扰动因素
        self.u_noise = None
        # communication noise amount 智能体的通信噪声数量（未理解
        self.c_noise = None
        # control range 智能体的控制范围（未理解
        self.u_range = 1.0
        # state 智能体的状态，该成员变量是智能状态
        self.state = AgentState()
        # action 智能体的动作
        self.action = Action()
        # script behavior to execute 智能体需要执行的指令，是否能被“外部策略”控制
        self.action_callback = None

# multi-agent world 多智能体环境的设定，设定了一些维数以及时间分辨率和力的特征
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)智能体与实体的列表
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality 实体的通信信道的维数（未理解
        self.dim_c = 0
        # position dimensionality 实体的物理位置的维数
        self.dim_p = 2
        # color dimensionality 实体的颜色的维数
        self.dim_color = 3
        # simulation timestep 仿真的时间分辨率
        self.dt = 0.1
        # physical damping 实体的物理抑制率（未理解
        self.damping = 0.25
        # contact response parameters 接触反应参数（未理解
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world 返回该环境中所有的智能体与地标列表
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies 返回所有可被环境之外的外部策略控制的智能体列表
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts 返回所有被环境内的命令控制的智能体的列表
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world 更新环境状态
    def step(self):
        # set actions for scripted agents 对按照脚本运行的智能体施加动作
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities 收集各实体收到的力的列表
        # 初始化各实体受到的力，这些力将会转换为加速度并最终体现在速度上
        p_force = [None] * len(self.entities)
        # apply agent physical controls 实体通过action对自己施加的力
        p_force = self.apply_action_force(p_force)
        # apply environment forces 环境对实体施加的力，在这里主要指的是碰撞
        p_force = self.apply_environment_force(p_force)
        # integrate physical state 更新每一个实体的位置与速度
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces 智能体通过action对自己施加的力
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                # 如果智能体是可以移动，那么其action就是智能体对自己施加的力
                # 这个过程可以考虑到智能体自身的噪声，因为智能体和人一样，并不能完全精确地迈出一步，或者输出一个精确的力
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # 将初始化为零的实体受到的力加上实体对自己施加的力并加上噪声
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities 智能体所收到的来自环境的力
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response 一个简单但并不高效的碰撞反应
        # 穷举所有的实体为A
        for a,entity_a in enumerate(self.entities):
            # 穷举所有的实体为B
            for b,entity_b in enumerate(self.entities):
                # 防止进行反复组合
                if(b <= a): continue
                # 计算两者碰撞后两个实体所受到的力分别为fa与fb
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                # 如果fa不为None：这意味着实体A是可以运动的
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    # 将环境施加给实体A的力增加到A所受的力的总和中去
                    p_force[a] = f_a + p_force[a]
                # 如果fb不为None：这意味着实体B是可以运动的
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    # 将环境施加给实体B的力增加到B所受的力的总和中去
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state 更新当前位置与当前运动速度
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            # 如果实体是不会动的，那么就没必要更新其当前位置与当前运动速度了
            if not entity.movable: continue
            # 速度会衰减，衰减率由self.damping控制
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                # 速度=原速度+加速度*时间=原速度+(实体所受力/实体质量)*时间
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                # 以上是两个维度的分别的速度，在这里计算的是速度的绝对值，也就是x轴速度与y轴速度的平方和
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    # 如果对实体设定了最大速度，则当速度超过最大速度时，施加下列的限制
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed

            # 位置=原位置+位置的变化量（位移）=原位置+实体的速度*时间分辨率
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now) 设定通信状态
        if agent.silent:
            # 不进行通信，此时传输的内容为0
            agent.state.c = np.zeros(self.dim_c)
        else:
            # 生成一个噪声
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            # 通信的内容就是实体在x轴y轴上对自己施加的力并加上一个噪声，当然噪声是可选的
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities 当两个实体相互接触时，计算这两个实体碰撞后各自实体受到的力
    def get_collision_force(self, entity_a, entity_b):
        # 如果两个实体中的一个是不可碰撞的，那么久没必要计算这两个实体碰撞产生的力了
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider 不是碰撞者
        # 如果这两个实体是同一个实体的话，一个实体不可能和它自己发生碰撞，所以也没必要计算这两个实体碰撞产生的力了
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities 计算两个实体之间的每一个维度上面的差值
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        # 通过勾股定理计算出两个实体时间的距离
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance 算出两个实体不发生碰撞的最小距离，该距离是两个实体的半径之和
        dist_min = entity_a.size + entity_b.size
        # softmax penetration 进入并渗透，动量交换吧应该
        k = self.contact_margin #接触反应函数
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None # 如果实体A可移动则在碰撞后获得一个力，否则不获得一个力
        force_b = -force if entity_b.movable else None # 如果实体B可移动则在碰撞后获得一个力，否则不获得一个力
        return [force_a, force_b]