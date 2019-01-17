function varargout = BasicGUI(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BasicGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @BasicGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before BasicGUI is made visible.
function BasicGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BasicGUI (see VARARGIN)

handles.output = hObject;
guidata(hObject, handles);

a = - 5;
b = 5;
N = 16384;
fe = N/(b-a);
te = 1/fe;

handles.xt = linspace(a,b-te,N);    % N intervalles -> il faut s'arrêter à b-T
handles.xf = linspace(-fe/2,fe/2-1/(b-a),N);    %idem


pulse = 2*pi*(1/10);   %oméga = 2*PI*f
deltaT = 0;         %décalage temporel du Dirac
deltaF = 5;         %décalage fréquentiel pour aliasing

handles.constant = ones(1,N);

for n=1:N
    handles.cos(1,n)= cos(pulse*((n-1)*te + a));  %cosinus
end

for n=1:N
    handles.sin(1,n)= sin(pulse*((n-1)*te + a));  %sinus
end

handles.dirac = zeros(1,N);
handles.dirac(1,(N/2+1)-deltaT)=1;  %dirac en deltaT

for n=1:N
    handles.expcmp(1,n)= exp(1i*(pulse*((n-1)*te + a)));  %exponentielle complexe
end

handles.rectangle = zeros(1,N);
for n=floor(N/2-0.1*N):floor(N/2+0.1*N)+1   %rectangle(0.1)
    handles.rectangle(1,n)= 1;
end

for n=1:N
    handles.expdecr(1,n)= exp(-pi*((n-1)*te + a)^2);  %exponentielle décroissante
end

handles.creneau = zeros(1,N);
act = -0.4;
for o=1:3
    for n=floor(N/2+act*N):floor(N/2+(act+0.2)*N)+1 %créneau
        handles.creneau(1,n)= 1;
    end
    act = act + 0.3;
end

for n=1:N
    handles.AntiAliasing(1,n)= sin(pulse*((n-1)*te + a)) + sin((pulse+deltaF)*((n-1)*te + a)) + sin((pulse+2*deltaF)*((n-1)*te + a)) + sin((pulse+3*deltaF)*((n-1)*te + a));
end

handles.currentData = handles.constant

% Choose default command line output for BasicGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = BasicGUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% --- Executes on selection change in MenuFonction.
function MenuFonction_Callback(hObject, eventdata, handles)

str = get(hObject, 'String');
val = get(hObject, 'Value');

switch str{val}
    case 'Constante'
        handles.currentData = handles.constant;
    case 'Cosinus'
        handles.currentData = handles.cos;
    case 'Sinus'
        handles.currentData = handles.sin;
    case 'Dirac'
        handles.currentData = handles.dirac;
    case 'Expontielle complexe'
        handles.currentData = handles.expcmp;
    case  'Rectangle de pas 0.2'
        handles.currentData = handles.rectangle;
    case  'Exponentielle décroissante'
        handles.currentData = handles.expdecr;
    case 'Signal créneau'
        handles.currentData = handles.creneau;
    case 'Aliasing'
        handles.currentData = handles.AntiAliasing;
end

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function MenuFonction_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MenuFonction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function figure1_SizeChangedFcn(hObject, eventdata, handles)


% --- Executes on button press in RB1.
function RB1_Callback(hObject, eventdata, handles)

% --- Executes on button press in RB2.
function RB2_Callback(hObject, eventdata, handles)


% --- Executes on button press in draw.
function draw_Callback(hObject, eventdata, handles)

    axes(handles.fig1);
    plot(handles.xt,real(handles.currentData));
    handles.F = tfour(handles.currentData);
    if handles.RB1.Value == 1
        minp = min(min(real(handles.F)),min(imag(handles.F)));
        maxp = max(max(real(handles.F)),max(imag(handles.F)));
        axes(handles.fig2);
        plot(handles.xf,real(handles.F));
        ylim([minp-0.1*(maxp-minp),maxp+0.1*(maxp-minp)]);
        axes(handles.fig3);
        plot(handles.xf,imag(handles.F));
        ylim([minp-0.1*(maxp-minp),maxp+0.1*(maxp-minp)]);
    elseif handles.RB2.Value == 1
        minp = min(abs(handles.F))
        maxp = max(abs(handles.F))
        axes(handles.fig2);
        plot(handles.xf,abs(handles.F));
        ylim([minp-0.1*(maxp-minp)-0.1,maxp+0.1*(maxp-minp)+0.1]);
        axes(handles.fig3);
        plot(handles.xf,angle(handles.F));
        ylim([-pi-0.5,pi+0.5]);
    end

    axes(handles.fig4);
    plot(handles.xt,real(tfourinv(handles.F)));


    guidata(hObject, handles);
