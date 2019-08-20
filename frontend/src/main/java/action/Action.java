package action;

import javax.servlet.http.HttpServletRequest;

public abstract class Action {
    
    public abstract boolean executer(HttpServletRequest request);
    
}
