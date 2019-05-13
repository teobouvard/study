package action;

import java.util.List;
import javax.servlet.http.HttpServletRequest;
import metier.data.Medium;
import static metier.service.Service.getListMedium;


public class ActionListerMediums extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
    
        List<Medium> listeMedium = getListMedium();
        request.setAttribute("liste-medium", listeMedium);
        return true;
    }

}
