package action;

import java.util.List;
import javax.servlet.http.HttpServletRequest;
import metier.data.Medium;
import static metier.service.Service.getListMedium;

public class ActionInfosMedium extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {

        List<Medium> listeMedium = getListMedium();
        Integer ID = Integer.parseInt(request.getParameter("id"));
        for(Medium m : listeMedium){
            if(m.getId() == ID){
                request.setAttribute("medium",m);
                request.setAttribute("statut",true);               
                return true;
            }
        }
        request.setAttribute("statut",false);   
        return true;
    }

}
