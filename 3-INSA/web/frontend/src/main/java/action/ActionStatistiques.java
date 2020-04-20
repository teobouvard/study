package action;

import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Employe;
import metier.data.Medium;
import static metier.service.Service.getListEmploye;
import static metier.service.Service.getListMedium;

public class ActionStatistiques extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {

        List<Medium> listeMedium = getListMedium();
        List<Employe> listeEmploye = getListEmploye();

        request.setAttribute("listeMedium", listeMedium);
        request.setAttribute("listeEmploye", listeEmploye);

        return true;
    }

}
